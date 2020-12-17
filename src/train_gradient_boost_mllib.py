import numpy as np
from time import time
from utils import preprocess_parallel
from operator import add

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline


if __name__ == '__main__':
    print("Starting")

    sc = SparkContext("local[40]", 'Parallel Bitcoin')
    sc.setLogLevel('warn')

    # preprocess the data
    sql_c, train_data, val_data, test_data = preprocess_parallel(sc)

    input_dim = len(train_data.select("features").first()[0])
    print(train_data.select("features").first())
    print("Input dim: %d" % input_dim)
    print("Train size: %d" % train_data.count())
    print("Validation size: %d" % val_data.count())
    print("Test size: %d" % test_data.count())

    steps = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
    depths = [3, 5, 7, 8]  # needs to be less than equal to 30

    min_mae = float('inf')
    # best_depth = 8
    # best_step = 0.5
    best_depth = 8
    best_step = 0.001

    score = {}
    time_taken = {}
    count = 0

    # find best hyperparameters using validation data
    for step in steps:
        for depth in depths:
            print("_" * 50)
            print(f"Step = {step}")
            print(f"Depth = {depth}")

            # create model
            now = time()
            dt = GBTRegressor(featuresCol='features', labelCol='Weighted_Price', stepSize=step, maxIter=20, maxDepth=depth)
            dt_model = dt.fit(train_data)
            end = time() - now

            # create evaluators
            dt_predictions = dt_model.transform(val_data)
            dt_mse_eval = RegressionEvaluator(labelCol="Weighted_Price", predictionCol="prediction", metricName="mse")
            dt_mae_eval = RegressionEvaluator(labelCol='Weighted_Price', predictionCol="prediction", metricName="mae")

            mse = dt_mse_eval.evaluate(dt_predictions)
            mae = dt_mae_eval.evaluate(dt_predictions)

            dt_predictions.select("features", 'Weighted_Price', 'prediction')

            # print stats
            print(f"Time to fit: {end}")
            print(f"Mean Squared Error (MSE) on test data = {mse}")
            print(f"Mean Absolute Error on test data = {mae}")

            if (mae < min_mae):
                min_mae = mae
                best_bins = bin
                best_depth = depth
                best_step = step

            score[count] = mse
            time_taken[count] = end
            count += 1
            print("\n" * 2)
            sql_c.clearCache()

    # print best hyperparameters
    print(f"Best Step: {best_step}")
    print(f"Best Depth: {best_depth}")

    # create model based on best hyperparameters
    now = time()
    dt = GBTRegressor(featuresCol='features', labelCol='Weighted_Price',  stepSize=best_step, maxIter=20, maxDepth=best_depth)
    dt_model = dt.fit(train_data)
    end = time() - now

    # test model on the test data
    dt_predictions = dt_model.transform(test_data)
    dt_mse_eval = RegressionEvaluator(labelCol="Weighted_Price", predictionCol="prediction", metricName="mse")
    dt_mae_eval = RegressionEvaluator(labelCol='Weighted_Price', predictionCol="prediction", metricName="mae")
    best_mse = dt_mse_eval.evaluate(dt_predictions)
    best_mae = dt_mae_eval.evaluate(dt_predictions)

    # print stats
    print(f"Time to Fit: {end}")
    print(f"Best Mean Squared Error (MSE) on Test Data = {best_mse}")
    print(f"Best Mean Absolute Error on Test Data = {best_mae}")

    pred_df = dt_predictions.select('prediction').toPandas()
    pred_df.rename(columns={"prediction": "Weighted_Price"}, inplace=True)
    # pred_df.to_csv('../data/predictions/gboost_mllib_y_pred.csv', index=False)
    print("Done")
