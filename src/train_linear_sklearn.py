import pandas as pd
import numpy as np
from time import time
from utils import preprocess_serial

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import linear_model

if __name__ == '__main__':

    print("Starting")

    # preprocess the data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_serial()

    print("X_train.shape, y_train.shape")
    print(X_train.shape, y_train.shape)
    print("X_test.shape, y_test.shape")
    print(X_test.shape, y_test.shape)

    # create model using the best hyperparameters found in the parallel implementation
    model = linear_model.Lasso(alpha=0)

    # fit data
    start = time()
    model.fit(X_train, y_train)
    dt = time() - start
    print("Time to fit: %f" % dt)

    # predict data
    y_pred = model.predict(X_test)

    # calculate stats
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MSE: {mse}, MAE: {mae}")

    # save predictions to csv
    pred_df = pd.DataFrame(y_pred, columns=['Weighted_Price'])
    pred_df.to_csv('../data/predictions/lr_sklearn_y_pred.csv', index=False)

    print("Done")
