from nltk.sentiment import SentimentIntensityAnalyzer
import json
import sys


def get_sentiment(content):
    sent = sia.polarity_scores(content)
    return sent['compound'], sent


if len(sys.argv) != 4:
    print('Usage: sentiment.py <twt|rdc|rdp> file.in file.out')

sia = SentimentIntensityAnalyzer()
file_in = sys.argv[2]
file_out = sys.argv[3]

if sys.argv[1].lower() == 'twt':
    r = open(file_in, 'r')
    item = json.load(r)
    r.close()
    results = []
    for sub in item:
        score, details = get_sentiment(sub['content'])
        results.append({
            'platform': 'twitter',
            'date': sub['date'],
            'replies': sub['replyCount'],
            'reposts': sub['retweetCount'],
            'likes': sub['likeCount'],
            'awards': sub['quoteCount'],
            'sentiment': score,
            'sentiment_details': details
        })
    with open(file_out, 'w') as w:
        w.write(json.dumps(results, indent=4))
elif sys.argv[1].lower() == 'rdc':
    r = open(file_in, 'r')
    item = json.load(r)
    r.close()
    results = []
    for sub in item:
        score, details = get_sentiment(sub['content'])
        results.append({
            'platform': 'reddit_comment',
            'date': sub['date'],
            'replies': '',
            'reposts': '',
            'likes': sub['score'],
            'awards': sub['total_awards_received'],
            'sentiment': score,
            'sentiment_details': details
        })
    with open(file_out, 'w') as w:
        w.write(json.dumps(results, indent=4))
elif sys.argv[1].lower() == 'rdp':
    r = open(file_in, 'r')
    item = json.load(r)
    r.close()
    results = []
    for sub in item:
        score, details = get_sentiment(sub['content'])
        results.append({
            'platform': 'reddit_post',
            'date': sub['date'],
            'replies': sub['num_comments'],
            'reposts': sub['num_crossposts'],
            'likes': sub['score'],
            'awards': sub['total_awards_received'],
            'sentiment': score,
            'sentiment_details': details
        })
    with open(file_out, 'w') as w:
        w.write(json.dumps(results, indent=4))


def score_coindesk(json_name):
    json_data = open(json_name, 'r')
    json_data = json.load(json_data)
    coin_list = list(json_data.keys())
    sia = SentimentIntensityAnalyzer()

    for coin in coin_list:
        news_set = set([(e['date'], e['title'] + ' ' + e['text']) for e in json_data[coin]])
        news_set_scored = []
        for news in news_set:
            if '2021' in news[0]:
                news_set_scored.append({
                    "date": news[0],
                    "text": news[1],
                    "sentiment": sia.polarity_scores(news[1])['compound'],
                    "sentiment_details": sia.polarity_scores(news[1])
                })
        json_object = json.dumps(news_set_scored, indent=4)
        with open(f"{coin}_coindesk_scored.json", 'w') as out:
            out.write(json_object)
