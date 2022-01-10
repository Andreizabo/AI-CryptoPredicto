import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from datetime import datetime
import csv


def dayz(year, dayz_modif):
    return [[year[i - j + dayz_modif] for j in range(dayz_modif)] for i in range(365 - dayz_modif)]


def parse_json_entry(entry, date_format):
    if 'likes' not in entry:
        return {
            "date": datetime.strptime(entry["date"], date_format),
            "sentiment": entry["sentiment"]
        }
    elif entry['replies'] == "":
        return {
            "date": datetime.strptime(entry["date"], date_format),
            "sentiment": entry["sentiment"] * (
                    entry['likes'] + entry['awards'] * 2)
        }
    else:
        return {
            "date": datetime.strptime(entry["date"], date_format),
            "sentiment": entry["sentiment"] * (
                        entry['likes'] + entry['replies'] * 1.2 + entry['reposts'] * 1.5 + entry['awards'] * 2
                        * (0.01 if entry['platform'] == 'twitter' else 1))
        }


def parse_json(json_to_be_returned, json_path, date_format):
    arr = json.load(open(json_path))

    for i in range(len(arr)):
        item = parse_json_entry(arr[i], date_format)
        key = item["date"].strftime("%Y-%m-%d")

        if not key in json_to_be_returned.keys():
            json_to_be_returned[key] = []

        json_to_be_returned[key].append(item["sentiment"])

    for key, value in json_to_be_returned.items():
        json_to_be_returned[key] = [sum(value)]

    return json_to_be_returned


def parse_prices(json_to_be_returned):
    with open("files/data.csv") as file:
        reader = csv.reader(file)

        first = True

        last_row = None

        for row in reader:
            if first:
                first = False
                continue

            trend = 0.5

            if last_row != None:
                trend = 0 if (float(row[4]) < float(last_row[4])) else 1

            last_row = row

            date = row[0]

            if date in json_to_be_returned:
                json_to_be_returned[date].append(trend)

    return {key: json_to_be_returned[key] for key in json_to_be_returned if len(json_to_be_returned[key]) >= 1}


def create_model():
    inputs = layers.Input(shape=(5, 4,))

    layer1 = layers.Dense(36, activation="relu", kernel_regularizer="l2")(inputs)
    layer2 = layers.Dense(48, activation="relu", kernel_regularizer="l2")(layer1)
    output = layers.Dense(1, activation="sigmoid")(layer2)

    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(loss='huber', optimizer='adam', metrics=['accuracy'])

    return model


def data_to_input(value):
    return value[0:-1]


def data_to_target(value):
    return value[-1]


def split_list(initial_list):
    list_a = initial_list[:int((len(initial_list) + 1) * 0.8)]
    list_b = initial_list[int((len(initial_list) + 1) * 0.8):]
    return list_a, list_b


def main():
    tweets = {}
    tweets = parse_json(tweets, "files/twt.json", "%Y-%m-%dT%H:%M:%S%z")
    # parse_prices(tweets)

    reddit_posts = {}
    reddit_posts = parse_json(reddit_posts, "files/rdp.json", "%Y-%m-%d %H:%M:%S")

    reddit_comments = {}
    reddit_comments = parse_json(reddit_comments, "files/rdc.json", "%Y-%m-%d %H:%M:%S")
    # parse_prices(reddit_comments)

    coindesk_news = {}
    coindesk_news = parse_json(coindesk_news, "files/ethereum_coindesk_scored.json", "%b %d, %Y")

    full_dict = {}
    for key in tweets:
        if key == '2021-05-11':
            # 2021-05-10 nu e in twitter :(
            key_a = '2021-05-10'
            full_dict[key_a] = [0]
            if key_a in reddit_posts:
                full_dict[key_a].append(reddit_posts[key_a][0])
            else:
                full_dict[key_a].append(0)
            if key_a in reddit_comments:
                full_dict[key_a].append(reddit_comments[key_a][0])
            else:
                full_dict[key_a].append(0)
            if key_a in coindesk_news:
                full_dict[key_a].append(coindesk_news[key_a][0])
            else:
                full_dict[key_a].append(0)
        full_dict[key] = tweets[key]
        if key in reddit_posts:
            full_dict[key].append(reddit_posts[key][0])
        else:
            full_dict[key].append(0)
        if key in reddit_comments:
            full_dict[key].append(reddit_comments[key][0])
        else:
            full_dict[key].append(0)
        if key in coindesk_news:
            full_dict[key].append(coindesk_news[key][0])
        else:
            full_dict[key].append(0)


    parse_prices(full_dict)

    train_input = dayz([data_to_input(value) for key, value in full_dict.items()], 5)
    train_target = dayz([data_to_target(value) for key, value in full_dict.items()], 5)

    train_input, test_input = split_list(train_input)
    train_target, test_target = split_list(train_target)

    model = create_model()
    model.fit(train_input, train_target, epochs=1000, batch_size=32)

    _, accuracy = model.evaluate(test_input, test_target)
    print('Accuracy: %.2f' % (accuracy * 100))


if __name__ == '__main__':
    main()
