from __future__ import absolute_import
from __future__ import print_function

# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import SGD
# from keras.utils import np_utils
# import keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import utils
from tensorflow import keras
import tensorflow as tf

from time import time

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd

from pyspark import SparkContext, SparkConf

# Define basic parameters
batch_size = 64
nb_classes = 10
epochs = 1

# Create Spark context
conf = SparkConf().setAppName('Mnist_Spark_MLP').setMaster('local[4]')
sc = SparkContext(conf=conf)

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = utils.to_categorical(y_train, nb_classes)
y_test = utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

metrics = ['accuracy']
sgd = SGD(lr=0.1)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=metrics)

print(model.optimizer)
print(keras.optimizers.serialize(model.optimizer))
print(f"Metrics before spark: {model.metrics}")

# Build RDD from numpy features and labels
rdd = to_simple_rdd(sc, x_train, y_train)

# Initialize SparkModel from Keras model and Spark context
spark_model = SparkModel(model, frequency='epoch', mode='asynchronous', metrics=metrics)

print(f"Metrics after spark: {spark_model.master_metrics}")

# Train Spark model
start = time()
spark_model.fit(rdd, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.1)
print(f"Fit took: {time() - start}")
# Evaluate Spark model by evaluating the underlying model
start = time()
score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
print(f"Fit took: {time() - start}")

print(f"Metrics final: {model.metrics_names}")

# model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.1)
# score = model.evaluate(x_test, y_test, verbose=2)

print('Test accuracy:', score[1])