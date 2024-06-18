import json
import os
import time
import uuid
from datetime import timedelta
from multiprocessing import Process

from flask import Flask, request, make_response
import jsonlines

import jieba
import wordcloud
import imageio

# -*- coding=utf-8
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import configparser

from collections import Counter

from predictor.Predictor import Predictor

from run_spider import RunSpider

import requests

app = Flask(__name__)
predictor = Predictor()

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


@app.route('/')
def hello_world():
    # 在新进程运行
    spider = RunSpider()
    subprocess = Process(target=spider.run, args=('tweet',))
    subprocess.start()
    return 'Hello World!'


@app.route('/sentiment_analysis', methods=['POST'])
def do_predict():
    data = json.loads(request.get_data())
    content = data.get('content')
    global predictor
    pred, label = predictor.do_predict(content)
    return {
        'pred': pred,
        'label': label
    }


@app.route('/single_tweet_analysis', methods=['POST'])
def tweet_analysis():
    data = json.loads(request.get_data())
    tweet_ids = []
    tweet_id = data.get('tweet_id')
    tweet_ids.append(tweet_id)
    # 在新进程运行
    spider = RunSpider()
    subprocess = Process(target=spider.run, args=('single_tweet', tweet_ids))
    subprocess.start()
    # 阻塞主进程至子进程执行完毕
    subprocess.join()
    file_name = f'spider_output/single_tweet_spider_' + tweet_id + '.jsonl'
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


@app.route('/user_analysis', methods=['POST'])
def user_analysis():
    data = json.loads(request.get_data())
    user_ids = []
    user_id = data.get('user_id')
    user_ids.append(user_id)
    # 在新进程运行
    spider = RunSpider()
    subprocess_1 = Process(target=spider.run, args=('tweet', user_ids))
    subprocess_1.start()
    subprocess_2 = Process(target=spider.run, args=('user', user_ids))
    subprocess_2.start()
    # 阻塞主进程至子进程执行完毕
    subprocess_1.join()
    subprocess_2.join()
    file_name_1 = f'spider_output/tweet_spider_' + user_id + '.jsonl'
    file_name_2 = f'spider_output/user_spider_' + user_id + '.jsonl'
    my_dict = {}
    with jsonlines.open(file_name_1) as f:
        num = 0
        for line in f:
            my_dict['tweet' + str(num)] = line
            num += 1
    with open(file_name_2, 'r', encoding='utf-8') as f:
        data = json.load(f)
    my_dict['tweetNum'] = num
    my_dict['user'] = data
    return my_dict


@app.route('/tweets_analysis', methods=['POST'])
def tweets_analysis():
    data = json.loads(request.get_data())
    contents = data.get('contents')
    global predictor
    labels = []
    all_tokens = []
    for content in contents:
        label, _ = predictor.do_predict(content)
        labels.append(label)

        # 分词
        tokens = jieba.lcut(content)
        all_tokens += tokens

    # 生成词云

    bg = imageio.v2.imread('wordcloud_img/weibo_logo.png')
    txt = " ".join(all_tokens)
    print(txt)
    w = wordcloud.WordCloud(width=512, height=512, background_color='white', colormap='Accent',
                            font_path="wordcloud_img/STSongti-SC-Regular.ttf", mask=bg)
    w.generate(txt)
    file_name = 'wordcloud_img/wordcloud_' + str(uuid.uuid1()) + '.jpeg';
    w.to_file(file_name)

    img_url = upload_to_tencent(file_name)
    print(img_url)
    return {
        'labels': labels,
        'imgUrl': img_url
    }


@app.route('/getImg', methods=['POST', 'GET'])
def get_img():
    path = request.args.get('imgPath')
    image_data = open(path, "rb").read()
    response = make_response(image_data)
    response.headers['Content-Type'] = 'image/png'  # 返回的内容类型必须修改
    return response


def upload_to_tencent(filepath):
    # -*- coding: UTF-8 -*-

    config = configparser.ConfigParser()
    config.read("tx_bucket_key.ini", encoding="utf-8")

    secret_id = config.get('section1', 'secret_id')
    secret_key = config.get('section1', 'secret_key')
    region = config.get('section1', 'region')
    print(region)
    config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
    client = CosS3Client(config)

    uploaded_file_path = 'img/' + str(uuid.uuid1()) + '.jpeg'

    client.upload_file(
        Bucket='blog-img-1308044888',
        LocalFilePath=filepath,  # 本地文件的路径
        Key=uploaded_file_path,  # 上传到桶之后的文件名
    )

    return 'https://blog-img-1308044888.cos.ap-shanghai.myqcloud.com/' + uploaded_file_path


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)