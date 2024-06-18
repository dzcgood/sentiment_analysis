import argparse

import numpy as np
import torch
from transformers import AutoTokenizer

from models.MyModel import MyModel


class Predictor():
    def __init__(self):
        self.args = self.parse_args()
        self.device = torch.device(self.args.device)
        self.model = MyModel()
        checkpoint = torch.load(self.args.model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        print('model_inited')
        self.model.eval()
        self.model.to(self.device)

    def parse_args(self):
        parser = argparse.ArgumentParser(description='WeiBo Sentiment Analysis -- Predict')
        parser.add_argument('--device', default='cpu', type=str, help='cpu or cuda')
        parser.add_argument('--model_name', default='bert-base-chinese', type=str,
                            help='huggingface transformer model name')
        parser.add_argument('--model_path', default=r'weights/best.pt', type=str, help='model path')
        parser.add_argument('--num_labels', default=6, type=int, help='fine-tune num labels')

        args = parser.parse_args(args=[])
        return args

    def do_predict(self, input):
        label = {0: 'happy', 1: 'angry', 2: 'sad', 3: 'fear', 4: 'surprise', 5: 'neutral'}
        tokenizer = AutoTokenizer.from_pretrained('weights/bert-base-chinese')
        # tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)

        token = tokenizer(input, padding='max_length', truncation=True, max_length=140)
        input_ids = torch.tensor(token['input_ids']).unsqueeze(0)
        attention_mask = torch.tensor(token['attention_mask']).unsqueeze(0)
        input_ids.to(self.device)
        attention_mask.to(self.device)
        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
        pred = np.argmax(output.logits.detach().cpu().numpy(), axis=1).tolist()[0]
        return pred, label[pred]



