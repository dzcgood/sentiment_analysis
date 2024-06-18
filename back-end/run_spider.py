#!/usr/bin/env python
# encoding: utf-8

import os
import sys

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings


from spiders.tweet import TweetSpider
from spiders.user import UserSpider
from spiders.single_tweet import SingleTweetSpider


class RunSpider:
    def __init__(self):
        os.environ['SCRAPY_SETTINGS_MODULE'] = 'settings'
        self.settings = get_project_settings()

    def run(self, spider_name, args):
        process = CrawlerProcess(self.settings)
        mode_to_spider = {
            'tweet': TweetSpider,
            'user': UserSpider,
            'single_tweet': SingleTweetSpider
        }

        if spider_name in ['tweet', 'user']:
            process.crawl(mode_to_spider[spider_name], user_ids=args)
            process.start()
        else:
            process.crawl(mode_to_spider[spider_name], tweet_ids=args)
            process.start()
