from pyspark.sql import SQLContext
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
import sklearn.preprocessing as sklp
from sklearn.model_selection import train_test_split
import pandas as pd


def preprocess_parallel(sc):
    """
    Preprocess data in parallel with pyspark. Splits data into train, validation, and test data.

    Args:
        sc (pyspark.SparkContext): spark context instance

    Returns:
        sql_c (pyspark.sql.SQLContext): SQL item
        train_data (pyspark.sql.DataFrame): pyspark dataframe for train
        val_data (pyspark.sql.DataFrame): pyspark dataframe for validation
        test_data (pyspark.sql.DataFrame): pyspark dataframe for test
    """
    sql_c = SQLContext(sc)
    df = sql_c.read.csv("../data/processed/bitstampUSD.csv", header=True, inferSchema=True)
    print(df.limit(5).toPandas())

    # create Pipeline and perform MinMaxScaling on features
    stages = []
    unscaled_features = df.columns
    unscaled_features.remove("Weighted_Price")
    unscaled_features.remove("Timestamp")
    unscaled_assembler = VectorAssembler(inputCols=unscaled_features, outputCol="unscaled_features")
    scaler = MinMaxScaler(inputCol="unscaled_features", outputCol="scaled_features")
    stages += [unscaled_assembler, scaler]

    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(df)
    df_transform = pipeline_model.transform(df)

    df_transform_fin = df_transform.select('scaled_features', 'Weighted_Price', 'Timestamp')
    df_transform_fin = df_transform_fin.withColumnRenamed("scaled_features", "features")

    train_data = df_transform_fin.filter(df_transform["Timestamp"] <= 1517740920).select('features', "Weighted_Price")  # 25-Jun-2018
    df_transform_pre = df_transform_fin.filter(df_transform["Timestamp"] > 1517740920)
    val_data = df_transform_pre.filter(df_transform["Timestamp"] <= 1545174480).select('features', "Weighted_Price")
    test_data = df_transform_pre.filter(df_transform["Timestamp"] > 1545174480).select('features', "Weighted_Price")
    return sql_c, train_data, val_data, test_data


def preprocess_serial():
    """
    Preprocess data serially. Splits data into train, validation, and test data.

    Returns:
        X_train (pd.DataFrame): train features dataframe
        X_val (pd.DataFrame): validation features dataframe
        X_test (pd.DataFrame): test features dataframe
        y_train (pd.DataFrame): train target dataframe
        y_val (pd.DataFrame): validation target dataframe
        y_test (pd.DataFrame): test target dataframe
    """
    df = pd.read_csv("../data/processed/bitstampUSD.csv")

    X = df.drop(["Weighted_Price"], axis=1)
    y = df["Weighted_Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=(2./3.), shuffle=False)

    sc = sklp.MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test
