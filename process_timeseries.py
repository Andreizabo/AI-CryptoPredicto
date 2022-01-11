import json

import pandas as pd
import datetime


def get_rsi(price_series, n=14):
    delta = price_series.diff(1)
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=n).mean()
    ma_down = down.rolling(window=n).mean()
    return 100 - (100 / (1 + ma_up / ma_down))


def get_moving_average(price_series, n=14):
    return price_series.rolling(window=n).mean()


def get_macd(price_series, n_fast=12, n_slow=26):
    ema_fast = price_series.ewm(span=n_fast, adjust=False).mean()
    ema_slow = price_series.ewm(span=n_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd, signal_line


def statistics_processing():
    data = pd.read_csv('./data/Binance_ETHUSDT_d.csv')
    data = data.iloc[::-1]
    data = data.drop(['unix', 'symbol', 'open', 'high', 'low', 'Volume ETH', 'Volume USDT'], axis=1)
    data['rsi'] = get_rsi(data['close'])
    data['moving_avg'] = get_moving_average(data['close'], n=14)
    data['macd'], data['signal_line'] = get_macd(data['close'])

    data['date'] = pd.to_datetime(data['date'])
    mask = (data['date'] >= '2021-1-1') & (data['date'] <= '2021-12-31')
    data = data.loc[mask]

    data.to_csv('./data/Binance_ETHUSDT_d_processed.csv', index=False)


def parse_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
        sentiments = {}
        start_date = datetime.datetime(2021, 1, 1)
        end_date = datetime.datetime(2021, 12, 31)
        while start_date <= end_date:
            sentiments[start_date.strftime('%Y-%m-%d')] = []
            start_date += datetime.timedelta(days=1)

        for item in data:
            date = item['date']
            if 'twitter' in path:
                date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S%z')
                score = item['sentiment'] * (item['likes'] + item['replies'] + item['awards'] + item['reposts'])
            elif 'post' in path:
                date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                score = item['sentiment'] * (
                            item['likes'] + item['replies'] + item['awards'] * 10 + item['reposts'] * 10)
            elif 'comm' in path:
                date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                score = item['sentiment'] * (item['likes'] + item['awards'] * 10)
            else:
                date = datetime.datetime.strptime(date, '%b %d, %Y')
                score = item['sentiment']
            sentiments[date.strftime('%Y-%m-%d')].append(score)

        to_return = []
        for value in sentiments.values():
            to_return.append(sum(value) / len(value) if len(value) > 0 else 0)
        # max_score = max(to_return)
        # to_return = [i / max_score for i in to_return]
        return to_return


if __name__ == '__main__':
    coindesk_data = parse_data('./data/ethereum_coindesk_scored.json')
    reddit_post_data = parse_data('./data/reddit_post.json')
    reddit_comment_data = parse_data('./data/reddit_comm.json')
    twitter_data = parse_data('./data/twitter.json')
    data = pd.read_csv('./data/Binance_ETHUSDT_d_processed.csv')
    data['coindesk'] = coindesk_data
    data['reddit_post'] = reddit_post_data
    data['reddit_comment'] = reddit_comment_data
    data['twitter'] = twitter_data
    data.to_csv('./data/Binance_ETHUSDT_d_processed_with_sentiment.csv', index=False)
