from pyspark import SparkConf, SparkContext
from utils import preprocess_parallel

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation

from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.optimizers import Adam
from elephas.spark_model import SparkModel
from time import time
import numpy as np
import pandas as pd

# Elephas for Deep Learning on Spark
from elephas.ml_model import ElephasEstimator


if __name__ == '__main__':
    # create spark context and read in processed data
    conf = SparkConf().setAppName("Parallel Bitcoin").setMaster('local[4]')
    sc = SparkContext(conf=conf)

    # split data into test/train datasets
    train_data, val_data, test_data = preprocess_parallel(sc)
    input_dim = len(train_data.select("features").first()[0])
    print(train_data.select("features").first())
    print("Input dim: %d" % input_dim)
    print("Train size: %d" % train_data.count())
    print("Test size: %d" % test_data.count())

    # create model object
    model = Sequential()
    model.add(LSTM(128, activation="sigmoid", input_shape=(1, input_dim)))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    metrics = [
        'MeanSquaredError',
        'MeanAbsoluteError'
    ]
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=metrics)
    print(model.summary())

    rdd = train_data.rdd.map(lambda x: (x[0].toArray().reshape(1, len(x[0])), x[1]))
    spark_model = SparkModel(model, frequency='epoch', mode='synchronous', metrics=metrics)
    start = time()
    spark_model.fit(rdd, epochs=1, batch_size=64, verbose=0, validation_split=0.1)
    fit_dt = time() - start
    print(f"Fit took: {fit_dt}")

    x_test = test_data.toPandas()['features']
    x_test = np.asarray(test_data.rdd.map(lambda x: x[0].toArray()).collect())
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    y_test = test_data.toPandas()["Weighted_Price"].to_numpy()
    y_test = y_test.reshape((len(y_test), 1, 1))
    print(f"X shape: {x_test.shape}, Y shape: {y_test.shape}")

    score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
    print(f"Test score: {score}")

    y_pred = spark_model.master_network.predict(x_test)
    pred_df = pd.DataFrame(y_pred, columns=['Weighted_Price'])
    pred_df.to_csv('../data/predictions/elephas_y_pred.csv', index=False)
    print("Done")
