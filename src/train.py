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

# Elephas for Deep Learning on Spark
from elephas.ml_model import ElephasEstimator

def dl_pipeline_fit_score_results(dl_pipeline, train_data, test_data, label):
    """
    """
    fit_dl_pipeline = dl_pipeline.fit(train_data)
    pred_train = fit_dl_pipeline.transform(train_data)
    pred_test = fit_dl_pipeline.transform(test_data)

    pnl_train = pred_train.select(label, "prediction")
    pnl_test = pred_test.select(label, "prediction")

    pred_and_label_train = pnl_train.rdd.map(lambda row: (row[label], row['prediction']))
    pred_and_label_test = pnl_test.rdd.map(lambda row: (row[label], row['prediction']))

    metrics_train = MulticlassMetrics(pred_and_label_train)
    metrics_test = MulticlassMetrics(pred_and_label_test)

    print("Training Data Accuracy: {}".format(round(metrics_train.precision(), 4)))
    print("Training Data Confusion Matrix")
    print(pnl_train.crosstab('label_index', 'prediction').toPandas())

    print("\nTest Data Accuracy: {}".format(round(metrics_test.precision(), 4)))
    print("Test Data Confusion Matrix")
    print(pnl_test.crosstab('label_index', 'prediction').toPandas())


if __name__ == '__main__':
    # create spark context and read in processed data
    conf = SparkConf().setAppName("Parallel Bitcoin").setMaster('local[6]')
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
    df_transform_fin = df_transform_fin.withColumnRenamed("scaled_features", "features") \
        .withColumnRenamed("Weighted_Price", "label")
    print(df_transform_fin.limit(5).toPandas())

    # split data into test/train datasets
    train_data = df_transform_fin.filter(df_transform_fin["Timestamp"] <= 1529899200).select('features', 'label')  # 25-Jun-2018
    test_data = df_transform_fin.filter(df_transform_fin["Timestamp"] > 1529899200).select('features', 'label')
    input_dim = len(train_data.select("features").first()[0])
    print(train_data.select("features").first())
    print("Input dim: %d" % input_dim)
    print("Train size: %d" % train_data.count())
    print("Test size: %d" % test_data.count())

    # create model object
    model = Sequential()
    model.add(LSTM(128, activation="sigmoid", input_shape=(input_dim,1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    metrics = ['accuracy']
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=metrics)
    print(model.summary())

    # model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=2)
    # x = (feats, label)
    rdd = train_data.rdd.map(lambda x: (x[0].toArray().reshape(len(x[0]),1), x[1]))
    spark_model = SparkModel(model, frequency='epoch', mode='asynchronous', metrics=metrics)
    spark_model.fit(rdd, epochs=100, batch_size=64, verbose=0, validation_split=0.1)
    # score = spark_model.master_network.evaluate()

    # # Create Estimator
    # optimizer_conf = optimizers.Adam(lr=0.01)
    # opt_conf = optimizers.serialize(optimizer_conf)
    #
    # estimator = ElephasEstimator()
    # estimator.set_keras_model_config(model.to_yaml())
    # estimator.set_num_workers(1)
    # estimator.set_epochs(100)
    # estimator.set_batch_size(64)
    # estimator.set_verbosity(1)
    # estimator.set_validation_split(0.10)
    # estimator.set_optimizer_config(opt_conf)
    # estimator.set_mode("synchronous")
    # estimator.set_loss("mean_squared_error")
    # estimator.set_metrics(['acc'])
    #
    # # Create learning pipeline and run
    # dl_pipeline = Pipeline(stages=[estimator])
    # dl_pipeline_fit_score_results(dl_pipeline, train_data, test_data, "label")

