import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

TRAIN_SPLIT = 290


def ts_train_test(all_data, time_steps, for_periods, add_tradecount=False, add_rsi=False, add_macd=False,
                  add_moving_avg=False, add_sentiments=False):
    ts_train = all_data['close'][:290]
    ts_test = all_data['close'][290:]
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    sc = MinMaxScaler(feature_range=(0, 1))
    ts_train_scaled = sc.fit_transform(np.array(ts_train).reshape(-1, 1))
    tradecount_scaled = sc.transform(np.array(all_data['tradecount']).reshape(-1, 1))
    rsi_scaled = sc.transform(np.array(all_data['rsi']).reshape(-1, 1))
    macd_scaled = sc.transform(np.array(all_data['macd']).reshape(-1, 1))
    macd_signal_scaled = sc.transform(np.array(all_data['signal_line']).reshape(-1, 1))
    moving_avg_scaled = sc.transform(np.array(all_data['moving_avg']).reshape(-1, 1))

    coindesk_scaled = sc.transform(np.array(all_data['coindesk']).reshape(-1, 1))
    reddit_posts_scaled = sc.transform(np.array(all_data['reddit_post']).reshape(-1, 1))
    reddit_comments_scaled = sc.transform(np.array(all_data['reddit_comment']).reshape(-1, 1))
    twitter_posts_scaled = sc.transform(np.array(all_data['twitter']).reshape(-1, 1))

    X_train = []
    y_train = []
    for i in range(time_steps, ts_train_len - for_periods):
        X_train.append(ts_train_scaled[i - time_steps:i])
        if add_tradecount:
            X_train[-1] = np.append(X_train[-1], tradecount_scaled[i - time_steps:i])
        if add_rsi:
            X_train[-1] = np.append(X_train[-1], rsi_scaled[i - time_steps:i])
        if add_macd:
            X_train[-1] = np.append(X_train[-1], macd_scaled[i - time_steps:i])
            X_train[-1] = np.append(X_train[-1], macd_signal_scaled[i - time_steps:i])
        if add_moving_avg:
            X_train[-1] = np.append(X_train[-1], moving_avg_scaled[i - time_steps:i])
        if add_sentiments:
            X_train[-1] = np.append(X_train[-1], coindesk_scaled[i - time_steps:i])
            X_train[-1] = np.append(X_train[-1], reddit_posts_scaled[i - time_steps:i])
            X_train[-1] = np.append(X_train[-1], reddit_comments_scaled[i - time_steps:i])
            X_train[-1] = np.append(X_train[-1], twitter_posts_scaled[i - time_steps:i])

        y_train.append(ts_train_scaled[i:i + for_periods])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    inputs = pd.concat((all_data["close"][:290], all_data["close"][290:]), axis=0).values
    inputs = inputs[len(inputs) - len(ts_test) - time_steps:]
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(time_steps, ts_test_len + time_steps - for_periods + 1):
        X_test.append(inputs[i - time_steps:i, 0])
        if add_tradecount:
            X_test[-1] = np.append(X_test[-1], tradecount_scaled[i - time_steps:i])
        if add_rsi:
            X_test[-1] = np.append(X_test[-1], rsi_scaled[i - time_steps:i])
        if add_macd:
            X_test[-1] = np.append(X_test[-1], macd_scaled[i - time_steps:i])
            X_test[-1] = np.append(X_test[-1], macd_signal_scaled[i - time_steps:i])
        if add_moving_avg:
            X_test[-1] = np.append(X_test[-1], moving_avg_scaled[i - time_steps:i])
        if add_sentiments:
            X_test[-1] = np.append(X_test[-1], coindesk_scaled[i - time_steps:i])
            X_test[-1] = np.append(X_test[-1], reddit_posts_scaled[i - time_steps:i])
            X_test[-1] = np.append(X_test[-1], reddit_comments_scaled[i - time_steps:i])
            X_test[-1] = np.append(X_test[-1], twitter_posts_scaled[i - time_steps:i])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, sc


def RNN(x_train, y_train, x_test, sc, dropout=True, neurons=32, epochs=100, batch_size=32):
    if dropout:
        model = keras.models.Sequential([
            keras.layers.SimpleRNN(neurons, return_sequences=True),
            keras.layers.SimpleRNN(neurons, return_sequences=True),
            keras.layers.SimpleRNN(neurons),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1)
        ])
    else:
        model = keras.models.Sequential([
            keras.layers.SimpleRNN(neurons, return_sequences=True),
            keras.layers.SimpleRNN(neurons, return_sequences=True),
            keras.layers.SimpleRNN(neurons),
            keras.layers.Dense(1)
        ])

    model.compile(loss="mse", optimizer="rmsprop")
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    predictions = model.predict(x_test)
    predictions = sc.inverse_transform(predictions)

    train_predict = model.predict(x_train)
    train_predict = sc.inverse_transform(train_predict)

    return model, predictions, train_predict


def LSTM(x_train, y_train, x_test, sc, dropout=True, neurons=32, epochs=100, batch_size=32):
    if dropout:
        model = keras.models.Sequential([
            keras.layers.LSTM(units=neurons, return_sequences=True, input_shape=(x_train.shape[1], 1),
                              activation='tanh'),
            keras.layers.LSTM(units=neurons, return_sequences=True, input_shape=(x_train.shape[1], 1),
                              activation='tanh'),
            keras.layers.LSTM(units=neurons, activation='tanh'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units=1)
        ])
    else:
        model = keras.models.Sequential([
            keras.layers.LSTM(units=neurons, return_sequences=True, input_shape=(x_train.shape[1], 1),
                              activation='tanh'),
            keras.layers.LSTM(units=neurons, return_sequences=True, input_shape=(x_train.shape[1], 1),
                              activation='tanh'),
            keras.layers.LSTM(units=neurons, activation='tanh'),
            keras.layers.Dense(units=1)
        ])

    model.compile(optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=True), loss='mse')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    predictions = model.predict(x_test)
    predictions = sc.inverse_transform(predictions)

    train_predict = model.predict(x_train)
    train_predict = sc.inverse_transform(train_predict)

    return model, predictions, train_predict


def GRU(x_train, y_train, x_test, sc, dropout=True, neurons=32, epochs=100, batch_size=32):
    if dropout:
        model = keras.models.Sequential([
            keras.layers.GRU(units=neurons, return_sequences=True, input_shape=(x_train.shape[1], 1),
                             activation='tanh'),
            keras.layers.GRU(units=neurons, return_sequences=True, input_shape=(x_train.shape[1], 1),
                             activation='tanh'),
            keras.layers.GRU(units=neurons, activation='tanh'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units=1)
        ])
    else:
        model = keras.models.Sequential([
            keras.layers.GRU(units=neurons, return_sequences=True, input_shape=(x_train.shape[1], 1),
                             activation='tanh'),
            keras.layers.GRU(units=neurons, return_sequences=True, input_shape=(x_train.shape[1], 1),
                             activation='tanh'),
            keras.layers.GRU(units=neurons, activation='tanh'),
            keras.layers.Dense(units=1)
        ])

    model.compile(optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=True), loss='mse')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    predictions = model.predict(x_test)
    predictions = sc.inverse_transform(predictions)

    train_predict = model.predict(x_train)
    train_predict = sc.inverse_transform(train_predict)

    return model, predictions, train_predict


def get_best_models():
    data = pd.read_csv('data/Binance_ETHUSDT_d_processed_with_sentiment.csv')
    data.set_index('date', inplace=True)
    x_train, y_train, x_test, sc = ts_train_test(data, 5, 1, add_sentiments=True)
    for model_fun in [RNN, LSTM, GRU]:
        model_name = model_fun.__name__
        if os.path.exists(f'models/best_{model_name}_sentiments.txt'):
            with open(f'models/best_{model_name}_sentiments.txt', 'r') as f:
                lines = f.readlines()
                best_mae = float(lines[1].split(':')[1].strip())
        else:
            best_mae = float('inf')
        for dropout in [True]:
            for neurons in [64]:
                for epochs in [1000, 1250, 1500]:
                    for batch_size in [16, 32, 64]:
                        for i in range(5):
                            print(
                                f'Training {model_name} with dropout={dropout}, neurons={neurons}, epochs={epochs}, batch_size={batch_size}')
                            model, predictions, train_predict = model_fun(x_train, y_train, x_test, sc, dropout,
                                                                          neurons, epochs, batch_size)
                            mae = np.abs(predictions.reshape(-1, ) - data['close'][290:]).mean()
                            print(f'MAE: {mae}')
                            if mae < best_mae:
                                best_mae = mae
                                best_model_name = f'{model_name}_{dropout}_{neurons}_{epochs}_{batch_size}'
                                model.save(f'models/best_{model_name}_sentiments.h5')
                                print('Found new best model:', best_model_name)
                                with open(f'models/best_{model_name}_sentiments.txt', 'w') as f:
                                    f.write(f'Best model: {best_model_name}\n')
                                    f.write(f'Best MAE: {best_mae}\n')


def main():
    # data = pd.read_csv('files/Binance_ETHUSDT_d_processed_with_sentiment.csv')
    # data.set_index('date', inplace=True)
    # X_train, y_train, X_test, sc = ts_train_test(data, 5, 1, add_sentiments=True)
    # model, predictions, train_predict = RNN(X_train, y_train, X_test, sc, epochs=1500, neurons=64)
    # plt.plot(data['close'][290:], label='Actual')
    # plt.plot(predictions, label='Predicted')
    # plt.legend()
    # plt.figure()
    # plt.plot(data['close'][:290], label='Actual')
    # plt.plot(train_predict, label='Train')
    # plt.legend()
    # plt.show()
    # print(np.abs(predictions.reshape(-1, ) - data['close'][290:]).mean())
    get_best_models()


if __name__ == '__main__':
    main()
