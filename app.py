import io

import flask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, request
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from flask import Response


class ModelHandler:
    TIME_STEPS = 5

    def __init__(self):
        # data = pd.read_csv('data/Binance_ETHUSDT_d.csv')
        data = pd.read_csv('data/Binance_ETHUSDT_d_processed_with_sentiment.csv')
        # data = data.iloc[::-1]
        data['date'] = pd.to_datetime(data['date'])
        self.data = data[['date', 'close', 'coindesk', 'reddit_post', 'reddit_comment', 'twitter']]
        self.data.set_index('date', inplace=True)
        self.models = {
            'LSTM': load_model('models/best_LSTM.h5'),
            'RNN': load_model('models/best_RNN.h5'),
            'GRU': load_model('models/best_GRU.h5'),
            'LSTM_sentiments': load_model('models/best_LSTM_sentiments.h5'),
            'RNN_sentiments': load_model('models/best_RNN_sentiments.h5'),
            'GRU_sentiments': load_model('models/best_GRU_sentiments.h5')
        }

    def get_data(self, start_date, end_date, sentiments=False):
        end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        data = self.data.loc[start_date:end_date]
        # data = self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date)]
        ts_train = data['close']
        ts_train_len = len(ts_train)

        sc = MinMaxScaler(feature_range=(0, 1))
        ts_train_scaled = sc.fit_transform(np.array(ts_train).reshape(-1, 1))
        coindesk_scaled = sc.transform(np.array(data['coindesk']).reshape(-1, 1))
        reddit_posts_scaled = sc.transform(np.array(data['reddit_post']).reshape(-1, 1))
        reddit_comments_scaled = sc.transform(np.array(data['reddit_comment']).reshape(-1, 1))
        twitter_posts_scaled = sc.transform(np.array(data['twitter']).reshape(-1, 1))

        # create training data of s samples and t time steps
        inputs = []
        for i in range(self.TIME_STEPS, ts_train_len - 1):
            inputs.append(ts_train_scaled[i - self.TIME_STEPS:i])
            if sentiments:
                inputs[-1] = np.append(inputs[-1], coindesk_scaled[i - self.TIME_STEPS:i])
                inputs[-1] = np.append(inputs[-1], reddit_posts_scaled[i - self.TIME_STEPS:i])
                inputs[-1] = np.append(inputs[-1], reddit_comments_scaled[i - self.TIME_STEPS:i])
                inputs[-1] = np.append(inputs[-1], twitter_posts_scaled[i - self.TIME_STEPS:i])

        inputs = np.array(inputs)

        # Reshaping X_train for efficient modelling
        inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))

        return data, inputs, sc

    def predict(self, model_name, start_date, end_date):
        start_date = pd.to_datetime(start_date)
        start_date -= pd.Timedelta(days=self.TIME_STEPS)
        end_date = pd.to_datetime(end_date)
        if model_name not in self.models.keys():
            raise Exception('Model not found')
        if start_date < self.data.index.min() or end_date > self.data.index.max() or start_date > end_date:
            raise Exception('Invalid date range')
        data, inputs, sc = self.get_data(start_date, end_date, sentiments='sentiments' in model_name)
        model = self.models[model_name]
        predictions = model.predict(inputs)
        predictions = sc.inverse_transform(predictions)
        data = data[-len(predictions):]
        data['prediction'] = predictions
        plt.figure()
        plt.plot(data['close'])
        plt.plot(data['prediction'])
        plt.legend(['Actual', 'Predicted'])
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        return buf.getvalue()


app = Flask(__name__, template_folder='./frontend', static_folder='./frontend')
model_handler = ModelHandler()


@app.route('/')
def hello():
    return flask.render_template('index.html')


@app.route('/background.svg')
def background_svg():
    return flask.send_from_directory('frontend', 'background.svg')


@app.route('/no_data.jpg')
def nodata_jpg():
    return flask.send_from_directory('frontend', 'no_data.jpg')


@app.route('/index.js')
def index_js():
    return flask.send_from_directory('frontend', 'index.js')


@app.route('/predict', methods=['POST'])
def rnn():
    try:
        start_date = request.json['start_date']
        end_date = request.json['end_date']
        model_name = request.json['model_name']
        image = model_handler.predict(model_name, start_date, end_date)
        return Response(image, mimetype='image/png')
    except Exception as e:
        return Response(str(e), status=500)


app.run(host='0.0.0.0', port=5000)
