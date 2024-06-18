import json
from scrapy import Spider
from scrapy.http import Request
from spiders.common import parse_tweet_info, parse_long_tweet




class SingleTweetSpider(Spider):
    """
    单一推文数据采集
    """
    name = "single_tweet_spider"
    base_url = "https://weibo.cn"

    def __init__(self, tweet_ids, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tweet_ids = tweet_ids

    # https: // weibo.cn / 2036070420 / FyGnYmrxE

    def start_requests(self):
        """
        爬虫入口
        """
        for tweet_id in self.tweet_ids:
            url = f"https://weibo.com/ajax/statuses/show?id={tweet_id}"
            yield Request(url, callback=self.parse, meta={'tweet_id': tweet_id})

    def parse(self, response, **kwargs):
        """
        网页解析
        """
        tweet = json.loads(response.text)
        item = parse_tweet_info(tweet)
        if item['isLongText']:
            url = "https://weibo.com/ajax/statuses/longtext?id=" + item['mblogid']
            yield Request(url, callback=parse_long_tweet, meta={'item': item})
        else:
            yield item

