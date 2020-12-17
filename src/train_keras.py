import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import preprocess_serial

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import tensorflow as tf

if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_serial()

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    print("X_train.shape, y_train.shape")
    print(X_train.shape, y_train.shape)
    print("X_test.shape, y_test.shape")
    print(X_test.shape, y_test.shape)

    model = Sequential()
    print(f"type: {type(model)}")
    model.add(LSTM(128, activation="sigmoid", input_shape=(1, 7)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    metrics = [
        'MeanSquaredError',
        'MeanAbsoluteError'
    ]
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=metrics)

    print("Model Summary:")
    print(model.summary())

    start = time()
    model.fit(X_train, y_train, epochs=1)
    dt = time() - start
    print("Time to fit: %f" % dt)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MSE: {mse}, MAE: {mae}")

    pred_df = pd.DataFrame(y_pred, columns=['Weighted_Price'])
    pred_df.to_csv('../data/predictions/keras_y_pred.csv', index=False)
    print("Done")
