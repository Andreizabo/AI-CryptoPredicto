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
