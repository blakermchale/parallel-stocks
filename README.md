# parallel-stocks
EECE5645 Parallel Processing Data Analytics Final Project  
Professor Ioannidis

## Installing

Log into the discovery cluster and run this while having a cluster checked out.

``` bash
conda create --name final python=3.7
conda activate final
conda install tensorflow-gpu
pip3 install tensorflow tensorflowonspark tensorflow_datasets --user
pip3 install -e elephas/
conda install -c conda-forge koalas
```

# Data 
[Huge Stock Market Dataset](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)

## Resources

[TensorFlowOnSpark](https://github.com/yahoo/TensorFlowOnSpark)  
[Discovery Keras](https://github.com/neu-spiral/Discovery-Cluster/wiki/keras)  
[Discovery GPU](https://github.com/neu-spiral/Discovery-Cluster/wiki/batch-mode)


