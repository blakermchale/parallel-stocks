from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics

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

# Elephas for Deep Learning on Spark
from elephas.ml_model import ElephasEstimator


if __name__ == '__main__':
    # create spark context and read in processed data
    conf = SparkConf().setAppName("Parallel Bitcoin").setMaster('local[20]')
    sc = SparkContext(conf=conf)
    sql_c = SQLContext(sc)
    df = sql_c.read.csv("../data/processed/bitstampUSD.csv", header=True, inferSchema=True)
    print(df.limit(5).toPandas())

    # create Pipeline and perform MinMaxScaling on features
    stages = []
    unscaled_features = df.columns
    unscaled_features.remove("Weighted_Price")
    # print(unscaled_features)
    unscaled_assembler = VectorAssembler(inputCols=unscaled_features, outputCol="unscaled_features")
    scaler = MinMaxScaler(inputCol="unscaled_features", outputCol="scaled_features")
    stages += [unscaled_assembler, scaler]

    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(df)
    df_transform = pipeline_model.transform(df)

    df_transform_fin = df_transform.select('scaled_features', 'Weighted_Price', 'Timestamp')
    df_transform_fin = df_transform_fin.withColumnRenamed("scaled_features", "features")
    print(df_transform_fin.limit(5).toPandas())

    # split data into test/train datasets
    train_data = df_transform_fin.filter(df_transform_fin["Timestamp"] <= 1529899200).select('features', "Weighted_Price")  # 25-Jun-2018
    test_data = df_transform_fin.filter(df_transform_fin["Timestamp"] > 1529899200).select('features', "Weighted_Price")
    input_dim = len(train_data.select("features").first()[0])
    print(train_data.select("features").first())
    print("Input dim: %d" % input_dim)
    print("Train size: %d" % train_data.count())
    print("Test size: %d" % test_data.count())

    # create model object
    model = Sequential()
    model.add(LSTM(128, activation="sigmoid", input_shape=(1, 7)))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    rdd = train_data.rdd.map(lambda x: (x[0].toArray().reshape(1, len(x[0])), x[1]))
    spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
    start = time()
    spark_model.fit(rdd, epochs=20, batch_size=64, verbose=0, validation_split=0.1)
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
    print("Done")
