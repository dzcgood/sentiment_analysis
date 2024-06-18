import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json


class MyDataset(Dataset):
    def __init__(self, path, tokenizer='bert-base-chinese'):
        # 数据
        self.data_list = self.read_json_file(path)
        # 分词器
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # 标签
        self.label = {'happy': 0, 'angry': 1, 'sad': 2, 'fear': 3, 'surprise': 4, 'neutral': 5}

    # 读取json格式的数据
    def read_json_file(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        return data_list

    # 重写__len__, __getitem__, collate_fn
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        content = data['content']
        label = self.label[data['label']]
        token = self.tokenizer(content, padding='max_length', truncation=True, max_length=140)
        input_ids = token['input_ids']
        token_type_ids = token['token_type_ids']
        attention_mask = token['attention_mask']
        return input_ids, token_type_ids, attention_mask, label

    @staticmethod
    def collate_fn(batch):
        input_ids, token_type_ids, attention_mask, label = list(zip(*batch))
        input_ids = torch.tensor(input_ids)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)
        label = torch.tensor(label)
        return input_ids, token_type_ids, attention_mask, label
