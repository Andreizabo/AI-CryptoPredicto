import math
import datetime
import os
import tqdm


class TwitterEater:
    def __init__(self):
        self.date = datetime.datetime.strptime('2021-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        self.end = datetime.datetime.strptime('2022-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        self.folder = 'E:/AI_data/twitter/'
        self.create_files()

    def create_files(self):
        try:
            p = open(os.path.join(self.folder, 'tweets.json'), 'x')
            p.close()
        except FileExistsError:
            pass
        try:
            p = open(os.path.join(self.folder, 'log.txt'), 'x')
            p.close()
        except FileExistsError:
            pass

    def eat(self):
        bar = tqdm.tqdm(range(math.floor(divmod((self.end - self.date).total_seconds(), 3600 * 24)[0])))
        while self.date < self.end:
            os.system(self.build_url(self.date))
            bar.desc = f"Finished iteration it={self.date.strftime('%Y-%m-%d')}"
            bar.update(1)
            self.date += datetime.timedelta(seconds=3600 * 24)

    def build_url(self, starting, duration=3600 * 24, limit=1000):
        return f'snscrape --jsonl --progress --max-results {limit} --since {starting.strftime("%Y-%m-%d")} twitter' \
               f'-search "(ethereum OR eth) min_replies:10 min_faves:10 min_retweets:10 ' \
               f'lang:en until:{(starting + datetime.timedelta(seconds=duration)).strftime("%Y-%m-%d")}" >> {os.path.join(self.folder, "tweets.json")} 2>> {os.path.join(self.folder, "log.txt")} '


if __name__ == '__main__':
    twt = TwitterEater()
    twt.eat()
