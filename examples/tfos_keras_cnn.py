import keras_cnn


if __name__ == '__main__':
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # absl_app.run(main)
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf
    from tensorflowonspark import TFCluster
    import argparse

    sc = SparkContext(conf=SparkConf().setAppName("mnist_ex"))
    executors = sc._conf.get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="number of records per batch", type=int, default=64)
    parser.add_argument("--buffer_size", help="size of shuffle buffer", type=int, default=1000)
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=3)
    parser.add_argument("--model_dir", help="path to save model/checkpoint", default="segmentation_model")
    parser.add_argument("--export_dir", help="path to export saved_model", default="segmentation_export")
    parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")

    args = parser.parse_args()
    print("args:", args)

    cluster = TFCluster.run(sc, keras_cnn.main, tf_args=args, num_executors=num_executors, num_ps=0, input_mode=TFCluster.InputMode.TENSORFLOW, master_node='chief')
    cluster.shutdown()
