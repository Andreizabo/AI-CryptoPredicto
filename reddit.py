import json
import math

import requests
import time
import datetime
import os
import tqdm


def eat_submission(submission, write, date):
    try:
        write.write(json.dumps({
            'num_comments': submission['num_comments'],
            'num_crossposts': submission['num_crossposts'],
            'score': submission['score'],
            'body': submission['selftext'],
            'title': submission['title'],
            'upvote_ratio': submission['upvote_ratio'],
            'total_awards_received': submission['total_awards_received'],
            'date': date.strftime('%Y-%m-%d %H:%M:%S')
        }, indent=4))
        write.write(',\n')
    except KeyError:
        # time.sleep(60)
        pass


def eat_comment(comment, write, date):
    try:
        write.write(json.dumps({
            'score': comment['score'],
            'body': comment['body'],
            'total_awards_received': comment['total_awards_received'],
            'date': date.strftime('%Y-%m-%d %H:%M:%S')
        }, indent=4))
        write.write(',\n')
    except KeyError:
        # time.sleep(60)
        pass


class RedditEater:
    def __init__(self, sub):
        self.sub = sub
        self.p_url = 'https://api.pushshift.io/reddit/submission/search/'
        self.c_url = 'https://api.pushshift.io/reddit/comment/search/'
        self.date = datetime.datetime.strptime('2021-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        self.end = datetime.datetime.strptime('2022-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        self.folder = 'E:/AI_data/reddit/'
        self.create_files()
        self.std_duration = 3600 * 24

    def create_files(self):
        p = open(os.path.join(self.folder, 'posts.json'), 'w')
        p.write('[\n')
        p.close()
        c = open(os.path.join(self.folder, 'comments.json'), 'w')
        c.write('[\n')
        c.close()

    def eat(self):
        sent = 0
        t_start = time.time()
        p = open(os.path.join(self.folder, 'posts.json'), 'a')
        c = open(os.path.join(self.folder, 'comments.json'), 'a')
        empty_bodies = 0
        bar = tqdm.tqdm(range(math.floor(divmod((self.end - self.date).total_seconds(), self.std_duration)[0])))
        while self.date < self.end:
            if sent >= 58:
                # print(f"Finished iteration eb={empty_bodies} it={self.date.strftime('%Y-%m-%d %H:%M:%S')}")
                p.close()
                c.close()
                while time.time() - t_start <= 60:
                    time.sleep(1)
                sent = 0
                t_start = time.time()
                p = open(os.path.join(self.folder, 'posts.json'), 'a')
                c = open(os.path.join(self.folder, 'comments.json'), 'a')
            try:
                res = json.loads(requests.get(self.build_url(self.p_url, self.date, duration=self.std_duration, limit=100)).content)
                for element in res["data"]:
                    eat_submission(element, p, self.date)
            except json.decoder.JSONDecodeError:
                empty_bodies += 1
            try:
                res = json.loads(requests.get(self.build_url(self.c_url, self.date, duration=self.std_duration, limit=100)).content)
                for element in res["data"]:
                    eat_comment(element, c, self.date)
            except json.decoder.JSONDecodeError:
                empty_bodies += 1
            bar.desc = f"Finished iteration eb={empty_bodies} it={self.date.strftime('%Y-%m-%d %H:%M:%S')}"
            self.date += datetime.timedelta(seconds=self.std_duration)
            sent += 2
            bar.update(1)
        p.write('{}\n]')
        p.close()
        c.write('{}\n]')
        c.close()

    def build_url(self, url, starting, duration=3600, limit=10):
        return f'{url}?after={starting}&before={starting + datetime.timedelta(seconds=duration)}&sort_type=score&sort=desc&subreddit={self.sub}&limit={limit}'


if __name__ == '__main__':
    rdet = RedditEater('ethereum')
    rdet.eat()
