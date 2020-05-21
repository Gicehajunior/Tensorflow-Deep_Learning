import tensorflow as tf
import tensorflow_datasets as tfd
import math
import logging
import model_train


model_train.logging

# our model incorporated with CNN, and maxPooling capabilities
def model():
    model = tf.keras.Squential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),
        tf.keras.layers.maxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
        tf.keras.layers.flatten()
        tf.keras.layers.maxPooling2D((2, 2), strides=2),
        tf.keras.layers.Dense(128, activation=tf.nn.relu)
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    
model = model()

# compiling our model now 
def compile_model(uncompiled_model):
    
    # optimization of our model 
    optimizable_model = uncompiled_model.compile(optimizer='adams', loss=tf.keras.SparseCategoricalCrossentropy(), metrics=['activation'])

    return optimized_model

trainable_model = compile_model(model)


# Training our model 
#caches our datasets 
#and shuffles the datasets 
# and applies the batch size of trials it can do during trains and tests
def train_model(untrained_model):
    
    BATCH_SIZE = 32
    training_data = tfd.train_dataset.cache().shuffle(tfd.explore()).batch(BATCH_SIZE)
    test_data = tfd.test_dataset.cache().batch(BATCH_SIZE)
    
    #to train our model to our data we should pass the datasets, epochs, math ceiling.
    # call the training function. by calling the method .fit()
    trained_model = untrained_model.fit(test_data, epochs=5, steps_per_epoch=math.ceil(tfd.explore(metadata)/BATCH_SIZE))
    
    return trained_model
    
trained_model = train_model(trainable_model)


#with the above model, now you can be able to predict the data supplied, i.e. tf mnists datasets