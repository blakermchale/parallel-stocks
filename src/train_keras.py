import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import tensorflow as tf

if __name__ == '__main__':
    df = pd.read_csv("../data/processed/bitstampUSD.csv")

    train_data = df.loc[df["Timestamp"] <= 1529899200]
    test_data = df.loc[df["Timestamp"] > 1529899200]
    print(train_data.shape)
    print(test_data.shape)

    train_set = train_data.values
    print(train_set.shape)

    sc = MinMaxScaler()
    y_train = train_set[:, -1]
    X_train = sc.fit_transform(train_set[:, :train_set.shape[1]-1])
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    test_set = test_data.values
    y_test = test_set[:, -1]
    X_test = sc.transform(test_set[:, :test_set.shape[1]-1])
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    model = Sequential()
    model.add(LSTM(128, activation="sigmoid", input_shape=(1, 7)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    metrics = [
        'MeanSquaredError',
        'MeanAbsoluteError'
    ]
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=metrics)
    print(model.summary())

    start = time()
    model.fit(X_train, y_train, epochs=1)
    dt = time() - start

    y_pred = model.predict(X_test)
    y_pred = sc.inverse_transform(y_pred)

    print("Time to fit: %f" % dt)
    mse = mean_squared_error(y_train, y_pred)



