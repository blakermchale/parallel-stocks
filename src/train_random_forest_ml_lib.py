import numpy as np
from time import time
from pyspark import SparkContext
from pyspark.sql import SQLContext
from operator import add

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline

if __name__ == '__main__':

    print("Starting")
    sc = SparkContext("local[40]", 'parallel_stocks')
    sc.setLogLevel('warn')

    sql_c = SQLContext(sc)
    df = sql_c.read.csv("../data/processed/bitstampUSD.csv", header=True, inferSchema=True)
    print(df.limit(5).toPandas())

    stages = []
    unscaled_features = df.columns
    unscaled_features.remove("Weighted_Price")

    unscaled_assembler = VectorAssembler(inputCols=unscaled_features, outputCol="unscaled_features")
    scaler = MinMaxScaler(inputCol="unscaled_features", outputCol="scaled_features")
    stages += [unscaled_assembler, scaler]

    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(df)
    df_transform = pipeline_model.transform(df)

    df_transform_fin = df_transform.select('scaled_features', 'Weighted_Price', 'Timestamp')
    df_transform_fin = df_transform_fin.withColumnRenamed("scaled_features", "features")
    print(df_transform_fin.limit(5).toPandas())

    # split data into test/train datasets on 25-Jun-2018
    train_data = df_transform_fin.filter(df_transform_fin["Timestamp"] <= 1529899200).select('features',
                                                                                             "Weighted_Price")
    test_data = df_transform_fin.filter(df_transform_fin["Timestamp"] > 1529899200).select('features', "Weighted_Price")
    input_dim = len(train_data.select("features").first()[0])
    print(train_data.select("features").first())
    print("Input dim: %d" % input_dim)
    print("Train size: %d" % train_data.count())
    print("Test size: %d" % test_data.count())


    trees = [10, 15, 20, 25, 30, 35]
    depths = [3, 5, 7]

    min_mae = float('inf')
    best_tree = 10
    best_depth = 3

    score = {}
    time_taken = {}
    count = 0

    # find best hyperparameters
    for tree in trees:
        for depth in depths:
            print("_" * 50)
            print("Number of trees = {}".format(tree))
            print("Depth = {}".format(depth))
            print("_" * 50)

            now = time()
            dt = RandomForestRegressor(featuresCol='features', labelCol='Weighted_Price', numTrees=tree, maxDepth=depth)
            dt_model = dt.fit(train_data)
            end = time() - now

            dt_predictions = dt_model.transform(test_data)
            dt_mse_eval = RegressionEvaluator(labelCol="Weighted_Price", predictionCol="prediction", metricName="mse")
            dt_mae_eval = RegressionEvaluator(labelCol='Weighted_Price', predictionCol="prediction", metricName="mae")

            mse = dt_mse_eval.evaluate(dt_predictions)
            mae = dt_mae_eval.evaluate(dt_predictions)

            dt_predictions.select("features", 'Weighted_Price', 'prediction').show(5)

            print(f"Time to fit: {end}")
            print(f"Mean Squared Error (MSE) on test data = {mse}")
            print(f"Mean Absolute Error on test data = {mae}")

            if (mae < min_mae):
                min_mae = mae
                best_tree = tree
                best_depth = depth

            score[count] = mse
            time_taken[count] = end
            count += 1
            print("\n" * 2)
            sql_c.clearCache()

    print(f"Best Number of Trees: {best_tree}")
    print(f"Best Depth: {best_depth}")

    now = time()
    dt = RandomForestRegressor(featuresCol='features', labelCol='Weighted_Price', numTrees=tree, maxDepth=depth)
    dt_model = dt.fit(train_data)
    end = time() - now

    dt_predictions = dt_model.transform(test_data)
    dt_mse_eval = RegressionEvaluator(labelCol="Weighted_Price", predictionCol="prediction", metricName="mse")
    dt_mae_eval = RegressionEvaluator(labelCol='Weighted_Price', predictionCol="prediction", metricName="mae")
    best_mse = dt_mse_eval.evaluate(dt_predictions)
    best_mae = dt_mae_eval.evaluate(dt_predictions)

    print(f"Time to Fit: {end}")
    print(f"Best Mean Squared Error (MSE) on Test Data = {best_mse}")
    print(f"Best Mean Absolute Error on Test Data = {best_mae}")

