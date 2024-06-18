import json
# 清洗微博文本
from harvesttext import HarvestText
import re
# 表情包 --> 中文含义
from emojiswitch import demojize
# 繁体转成简体、大写转成小写、全角转半角等
import pyhanlp
# 数据增强，暂时没用到
# import jionlp as jio
from tqdm import trange
import os
import argparse


# 读取json文件
def read_json_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('file {} finished reading'.format(file))
    return data


# 保存json文件
def save_json_file(data, file, indent=1):
    with open(file, 'w', encoding='utf-8') as f:
        # 不使用默认的ASCII编码
        f.write(json.dumps(data, indent=indent, ensure_ascii=False))
    print('file {} finished writing'.format(file))


# 去除微博文本中的url链接
def remove_url(src_str):
    # 当成多行处理
    return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', src_str, flags=re.MULTILINE)


# 数据增强
# def data_enhance(src_str):
#     # 同音词替换
#     return jio.homophone_substitution(src_str)


# 主逻辑
def clean_data(file, save_dir):
    filename = file.split('/')[-1]
    data = read_json_file(file)
    cleaned_data = []
    CharTable = pyhanlp.JClass('com.hankcs.hanlp.dictionary.other.CharTable')
    ht = HarvestText()
    for i in trange(len(data)):
        # 繁体转成简体、大写转成小写、全角转半角等
        # 去除url
        # 表情包 --> 中文
        content = demojize(remove_url(CharTable.convert(data[i]['content'])), delimiters=("[", "]"), lang="zh")
        # 清洗微博文本，如@，// 等
        cleaned_content = ht.clean_text(content, emoji=False, t2s=True)
        # 如果content为空，则舍弃
        if 'train' in file and (not content or not cleaned_content):
            continue
        if 'eval' in file or 'test' in file:
            # 'eval'和'test'数据不考虑数据增强
            cleaned_data.append({'content': cleaned_content, 'label': data[i]['label']})
        else:
            # 'train'数据考虑进行数据增强
            cleaned_data.append({'content': cleaned_content, 'label': data[i]['label']})
            # enhanced_data = data_enhance(data[i]['content'])
            # for j in enhanced_data:
            #     cleaned_data.append({'content': j, 'label': data[i]['label']})
    # 保存数据
    save_json_file(cleaned_data, os.path.join(save_dir, filename))

    def parse_args():
        parser = argparse.ArgumentParser(description='WeiBo Sentiment Analysis -- Clean Data')
        parser.add_argument('--absolute_path', default='/kaggle/input/smp2020', type=str,
                            help='system path')
        args = parser.parse_args(args=[])
        return args

    if __name__ == '__main__':
        args = parse_args()
        save_dir = args.absolute_path + '/clean'
        # train
        clean_data(args.absolute_path + '/work/train/virus_train.txt', save_dir)
        clean_data(args.absolute_path + '/work/train/usual_train.txt', save_dir)
        # eval
        clean_data(args.absolute_path + '/work/eval/virus_eval_labeled.txt', save_dir)
        clean_data(args.absolute_path + '/work/eval/usual_eval_labeled.txt', save_dir)
        # test
        clean_data(args.absolute_path + '/work/test/virus_test_labeled.txt', save_dir)
        clean_data(args.absolute_path + '/work/test/usual_test_labeled.txt', save_dir)
