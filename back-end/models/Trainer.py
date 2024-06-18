import argparse
import os
import random
import shutil

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from torch.cuda import amp
from loguru import logger
from tqdm import tqdm

from models.AvgUtils import AvgUtils
from models.MyDataSet import MyDataset
from models.MyModel import MyModel


class Trainer():
    def __init__(self, train_dataloader, model, lr, epochs, output_dir, val_dataloader=None, save_model_iter=5):
        self.device = torch.device('cuda:0')
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        # 　训练数据有几个batch
        self.n_batch = len(train_dataloader)
        self.model = model.to(self.device)
        self.lr = lr
        # 优化器，bert和lstm分别使用不同的lr
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if 'bert' in n]},
            {'params': [p for n, p in model.named_parameters() if 'bert' not in n],
             'lr': 2e-5}]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

        # 一共有几个step
        self.num_training_steps = self.n_batch * epochs
        # 对学习率更新调整，即在前3个epoch学习率从0线性增加，之后的每个epoch学习率从lr线性降低到0
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_training_steps=self.num_training_steps,
                                                         num_warmup_steps=3 * self.n_batch)
        # 自动混合精度
        self.scaler = amp.GradScaler(enabled=True)
        self.epochs = epochs
        self.start_epoch = 0
        self.global_step = 0
        self.save_model_iter = save_model_iter
        self.output_dir = output_dir

        # 日志初始化
        self._init_logger()
        # tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

        # 训练过程中的最好结果
        self.best_result = {'val_loss': 0, 'val_acc': 0, 'val_f1_score': 0}

    # 日志初始化，创建日志文件
    def _init_logger(self):
        log_file_name = 'train_{time}.log'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        log_path = os.path.join(self.output_dir, log_file_name)
        logger.add(log_path, rotation='1 week', retention='30 days', enqueue=True)

    # 每个epoch的训练逻辑
    def on_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(enumerate(self.train_dataloader), total=self.n_batch)

        train_loss = AvgUtils('Loss', ':6.3f')
        acc = AvgUtils('Acc', ':6.3f')
        f1 = AvgUtils('F1_score', ':6.3f')

        for index, (input_ids, token_type_ids, attention_mask, label) in pbar:
            self.global_step += 1
            output = model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device),
                           labels=label.to(self.device))
            loss = output.loss
            logits = output.logits

            # 自动混合精度
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

            # 预测值
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            train_f1_score = f1_score(y_true=label.cpu().numpy(), y_pred=preds, average='macro')
            train_acc = accuracy_score(y_true=label.cpu().numpy(), y_pred=preds)

            # 更新平均loss，平均acc，平均f1
            train_loss.update(loss.detach().cpu().numpy().tolist())
            acc.update(train_acc)
            f1.update(train_f1_score)

            # tqdm，设置精确到小数点后三位以及显示的内容
            pbar.set_description(f'{train_loss.avg: .3f}  {acc.avg: .3f}  {f1.avg: .3f}')
            pbar.set_postfix({'epoch': str(epoch) + '/' + str(self.epochs),
                              'bert_lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                              'other_lr': self.optimizer.state_dict()['param_groups'][1]['lr']})

            # tensorboard
            self.writer.add_scalar('Train/loss', train_loss.avg, self.global_step)
            self.writer.add_scalar('Train/acc', acc.avg, self.global_step)
            self.writer.add_scalar('Train/f1_score', f1.avg, self.global_step)
            self.writer.add_scalar('Train/bert_lr', self.optimizer.state_dict()['param_groups'][0]['lr'],
                                   self.global_step)
            self.writer.add_scalar('Train/other_lr', self.optimizer.state_dict()['param_groups'][1]['lr'],
                                   self.global_step)

        # 输出日志信息
        logger.info(f'{train_loss.avg: .3f}  {acc.avg: .3f}  {f1.avg: .3f}')

    # 每个epoch结束的时候做的事情
    def after_epoch(self, epoch):
        is_best_result = False
        cur_result = None
        if self.val_dataloader is not None:
            with torch.no_grad():
                # 跑一遍测试集
                val_loss, val_acc, val_f1_score = self.evaluate()

            self.writer.add_scalar('Val/loss', val_loss, epoch)
            self.writer.add_scalar('Val/acc', val_acc, epoch)
            self.writer.add_scalar('Val/f1_score', val_f1_score, epoch)

            cur_result = {'val_loss': val_loss, 'val_acc': val_acc, 'val_f1_score': val_f1_score}
            logger.info(f'complete {epoch} epochs, val result: {cur_result}')

            if val_f1_score > self.best_result['val_f1_score']:
                is_best_result = True
                self.best_result = cur_result

        # 如果是最好情况就保存
        if is_best_result:
            self.save_checkpoint(epoch, cur_result, True)
        # 每隔n个eppch也保存
        elif epoch % self.save_model_iter == 0:
            self.save_checkpoint(epoch, cur_result)
        else:
            pass

    # 保存模型
    def save_checkpoint(self, epoch, cur_result, save_best=False):

        state_dict = self.model.state_dict()
        state = {
            'epoch': epoch,
            'cur_result': cur_result,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        name = f'{epoch}.pt'
        path = os.path.join(self.output_dir, name)
        torch.save(state, path, _use_new_zipfile_serialization=False)
        # 如果是目前最好结果，就拷贝一份到best.pt
        if save_best:
            shutil.copy(path, os.path.join(self.output_dir, 'best.pt'))
            logger.info(f'Saving current best: {path}')
        else:
            logger.info(f'Saving checkpoint: {path}')

    # 训练模型的逻辑
    def train(self):
        logger.info('===============================START===============================')
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            print('\n loss    acc    f1_score')
            self.on_epoch(epoch)
            self.after_epoch(epoch)
        logger.info(f'complete the training, best result: {self.best_result}')
        logger.info('===============================END===============================')

    # val
    def evaluate(self):
        self.model.eval()
        pbar = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader))

        preds = []
        labels = []
        losses = []
        for index, (input_ids, token_type_ids, attention_mask, label) in pbar:
            output = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device),
                                labels=label.to(self.device))

            loss = output.loss
            logits = output.logits
            losses.append(loss.detach().cpu().numpy().tolist())
            preds.extend(np.argmax(logits.detach().cpu().numpy(), axis=1).tolist())
            labels.extend(label.cpu().numpy().tolist())
        val_f1_score = f1_score(y_true=np.array(labels), y_pred=np.array(preds), average='macro')
        val_acc = accuracy_score(y_true=np.array(labels), y_pred=np.array(preds))
        val_loss = np.array(losses).mean()
        return val_loss, val_acc, val_f1_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    # 描述
    parser = argparse.ArgumentParser(description='WeiBo Sentiment Analysis -- Train')
    # 与训练模型 Bert-base-Chinese
    parser.add_argument('--model_name', default='bert-base-chinese', type=str,
                        help='huggingface transformer model name')
    # 标签种类
    parser.add_argument('--num_labels', default=6, type=int, help='num labels')
    # 训练数据集路径
    parser.add_argument('--train_data_path', default='/kaggle/input/smp2020/clean/usual_train.txt', type=str,
                        help='train data path')
    # val数据集路径
    parser.add_argument('--val_data_path', default='/kaggle/input/smp2020/clean/usual_eval_labeled.txt', type=str,
                        help='train data path')
    # batch size
    parser.add_argument('--batch_size', default=32, type=int, help='train and validation batch size')

    parser.add_argument('--dataloader_num_workors', default=2, type=int, help='pytorch dataloader num workers')
    # 学习率
    parser.add_argument('--lr', default=1e-5, type=float, help='train learning rate')
    # epoch数
    parser.add_argument('--epochs', default=10, type=int, help='train epochs')
    # 模型保存路径
    parser.add_argument('--output_dir', default='/kaggle/working/', type=str, help='save dir')
    # 每隔几个epoch存一次
    parser.add_argument('--save_model_iter', default=5, type=int, help='save model num epochs on training')
    args = parser.parse_args(args=[])
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    # 就用今天的日期吧
    set_seed(20230304)
    model = MyModel()

    # train --> dataset, dataloader
    train_dataset = MyDataset(args.train_data_path, args.model_name)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.dataloader_num_workors,
                                  collate_fn=MyDataset.collate_fn)

    # val --> dataset, dataloader
    val_dataset = MyDataset(args.val_data_path, args.model_name)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.dataloader_num_workors,
                                collate_fn=MyDataset.collate_fn, drop_last=False)

    trainer = Trainer(train_dataloader=train_dataloader, model=model, lr=args.lr, epochs=args.epochs,
                      output_dir=args.output_dir, save_model_iter=args.save_model_iter, val_dataloader=val_dataloader)
    trainer.train()