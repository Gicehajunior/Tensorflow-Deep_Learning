import tensorflow as tf
import matplotlib.pylot as plot
import logging
import numpy as np
import math
import tensorflow_datasets

# global declaration of variables
trained_data_array = ""
testing_data_array = ""


# setup the layers of our model
def ml_model():
    relu = tf.nn.relu
    softmax = tf.nn.softmax

    model = tf.keras.models.Sequential([
        tf.keras.layers.flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.softmax),
        
    ])

    return model
model = ml_model()

# compile the model for build by optimization
def optimizer(unoptimized_model):
    model = model.compile(optimizer='adams', loss=tf.keras.loss.SparseCategoricalCrosstrophy(), metrics=['metrics'])
    
    optimized_model = model
    
    return optimized_model

optimized_model = optimizer(model)

#Training the model to do something
def train_model(fresh_optimized_model):
    # batch of 32 images to be used
    BATCH_SIZE = 32
    
    global trained_data_array
    global testing_data_array
    
    # format the datasets to be ready for training
    trained_data_array = tensorflow_datasets.train_dataset.cache().repeat().shuffle(tensorflow_datasets.explore()).batch(BATCH_SIZE)
    testing_data_array = tensorflow_datasets.test_dataset.cache().batch(BATCH_SIZE)
    
    ready_model = fresh_optimized_model.fit(trained_data_array, epochs=5, steps_per_epoch=math.ceil(tensorflow_datasets.explore()/BATCH_SIZE))

    return ready_model

ready_model = train_model(optimized_model)
