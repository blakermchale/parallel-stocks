import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.ensemble import GradientBoostingRegressor

if __name__ == '__main__':
    df = pd.read_csv("../data/processed/bitstampUSD.csv")

    print("Data Head:")
    print(df.head())

    # split data into train and test
    train_data = df.loc[df["Timestamp"] <= 1529899200]
    test_data = df.loc[df["Timestamp"] > 1529899200]
    print(f"Train Data Shape: {train_data.shape}")
    print(f"Test Data Shape: {test_data.shape}")

    train_set = train_data.values
    print(f"Train Set Shape: {train_set.shape}")

    # preprocess data
    sc = MinMaxScaler()
    y_train = train_set[:, -1]
    X_train = sc.fit_transform(train_set[:, :train_set.shape[1] - 1])

    test_set = test_data.values
    y_test = test_set[:, -1]
    X_test = sc.transform(test_set[:, :test_set.shape[1] - 1])

    print("X_train.shape, y_train.shape")
    print(X_train.shape, y_train.shape)
    print("X_test.shape, y_test.shape")
    print(X_test.shape, y_test.shape)

    # create XGBoost model

    # Setting SEED for reproducibility
    SEED = 1
    model = GradientBoostingRegressor(n_estimators = 100, max_depth = 5, random_state = SEED)

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
    pred_df.to_csv('../data/predictions/gradient_boost_y_pred.csv', index=False)
    print("Done")
