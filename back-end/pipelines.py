# -*- coding: utf-8 -*-
import datetime
import json
import os.path
import time


class JsonWriterPipeline(object):
    """
    写入json文件的pipline
    """

    def __init__(self):
        self.file = None
        if not os.path.exists('spider_output'):
            os.mkdir('spider_output')

    def process_item(self, item, spider):
        """
        处理item
        """
        if not self.file:
            if spider.name in ['tweet_spider', 'user_spider']:
                file_name = spider.name + "_" + spider.user_ids[0] + '.jsonl'
            else:
                file_name = spider.name + "_" + spider.tweet_ids[0] + '.jsonl'
            self.file = open(f'spider_output/{file_name}', 'wt', encoding='utf-8')
        item['crawl_time'] = int(time.time())
        line = json.dumps(dict(item), ensure_ascii=False) + "\n"
        self.file.write(line)
        self.file.flush()
        return item
