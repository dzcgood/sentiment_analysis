
import json
from scrapy import Spider
from scrapy.http import Request

from spiders.common import parse_tweet_info, parse_long_tweet


class TweetSpider(Spider):
    """
    用户推文数据采集
    """
    name = "tweet_spider"
    base_url = "https://weibo.cn"

    def __init__(self, user_ids, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_ids = user_ids
        self.tweet_num = 0

    def start_requests(self):
        """
        爬虫入口
        """
        # 这里user_ids可替换成实际待采集的数据
        for user_id in self.user_ids:
            url = f"https://weibo.com/ajax/statuses/mymblog?uid={user_id}&page=1"
            yield Request(url, callback=self.parse, meta={'user_id': user_id, 'page_num': 1})

    def parse(self, response, **kwargs):
        """
        网页解析
        """
        data = json.loads(response.text)
        tweets = data['data']['list']
        print(tweets)
        for tweet in tweets:
            self.tweet_num += 1
            if self.tweet_num <= 10:
                item = parse_tweet_info(tweet)
                if item['isLongText']:
                    url = "https://weibo.com/ajax/statuses/longtext?id=" + item['mblogid']
                    yield Request(url, callback=parse_long_tweet, meta={'item': item})
                else:
                    yield item
            else:
                break
        if self.tweet_num <= 10 and tweets:
            user_id, page_num = response.meta['user_id'], response.meta['page_num']
            page_num += 1
            url = f"https://weibo.com/ajax/statuses/mymblog?uid={user_id}&page={page_num}"
            yield Request(url, callback=self.parse, meta={'user_id': user_id, 'page_num': page_num})
