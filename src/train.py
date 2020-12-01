from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline


if __name__ == '__main__':
    conf = SparkConf().setAppName("Parallel Bitcoin").setMaster('local[6]')
    sc = SparkContext(conf=conf)
    sql_c = SQLContext(sc)

    df = sql_c.read.csv("../data/processed/bitstampUSD.csv", header=True, inferSchema=True)

    df.printSchema()
    print(df.limit(5).toPandas())

    stages = []
    unscaled_features = df.columns
    unscaled_features.remove("Weighted_Price")
    print(unscaled_features)
    unscaled_assembler = VectorAssembler(inputCols=unscaled_features, outputCol="unscaled_features")
    scaler = MinMaxScaler(inputCol="unscaled_features", outputCol="scaled_features")

    stages += [unscaled_assembler, scaler]
    print(stages)

    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(df)
    df_transform = pipeline_model.transform(df)

    df_transform_fin = df_transform.select('scaled_features', 'Weighted_Price')

    print(df_transform_fin.limit(5).toPandas())





