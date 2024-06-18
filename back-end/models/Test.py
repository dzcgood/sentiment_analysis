import argparse

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader

from models.MyDataSet import MyDataset
from models.MyModel import MyModel
from loguru import logger
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='WeiBo Sentiment Analysis -- Test')
    parser.add_argument('--model_name', default='bert-base-chinese', type=str,
                        help='huggingface transformer model name')
    parser.add_argument('--model_path', default=r'../weights/best.pt', type=str, help='model path')
    parser.add_argument('--num_labels', default=6, type=int, help='fine-tune num labels')
    parser.add_argument('--test_data_path', default=r'../smp2020/clean/usual_test_labeled.txt', type=str,
                        help='test data path')
    parser.add_argument('--batch_size', default=32, type=int, help='train and validation batch size')
    parser.add_argument('--dataloader_num_workors', default=2, type=int, help='pytorch dataloader num workers')
    args = parser.parse_args(args=[])
    return args


def main():
    args = parse_args()
    device = torch.device('cuda:0')
    model = MyModel()

    # 加载模型
    logger.info(f'Loading checkpoint: {args.model_path} ...')
    checkpoint = torch.load(args.model_path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=True)

    logger.info(f'missing_keys: {missing_keys}\n'
                f'===================================================================\n')
    logger.info(f'unexpected_keys: {unexpected_keys}\n'
                f'===================================================================\n')

    # test --> dataset, dataloader
    test_dataset = MyDataset(args.test_data_path, args.model_name)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.dataloader_num_workors,
                                 collate_fn=MyDataset.collate_fn, drop_last=False)

    model.eval()
    model.to(device)
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

    preds = []
    labels = []
    losses = []
    for index, (input_ids, token_type_ids, attention_mask, label) in pbar:
        with torch.no_grad():
            output = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                           labels=label.to(device))

        loss = output.loss
        logits = output.logits
        losses.append(loss.detach().cpu().numpy().tolist())
        preds.extend(np.argmax(logits.detach().cpu().numpy(), axis=1).tolist())
        print(np.argmax(logits.detach().cpu().numpy(), axis=1).tolist())
        labels.extend(label.cpu().numpy().tolist())

    test_f1_score = f1_score(y_true=np.array(labels), y_pred=np.array(preds), average='macro')
    test_acc = accuracy_score(y_true=np.array(labels), y_pred=np.array(preds))
    test_loss = np.array(losses).mean()

    logger.info(f'test_f1_score: {test_f1_score: .3f} \n '
                f'test_acc: {test_acc: .3f} \n'
                f'test_loss: {test_loss: .3f}')


if __name__ == '__main__':
    main()