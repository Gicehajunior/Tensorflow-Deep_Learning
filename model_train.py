import tensorflow as tf
import numpy as np
import logging

history = ""

#telling the tensorflow to log the errors if there is
def logging():
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)
    
logging()

#MODEL TRAINING

#instantiate the input and output
celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)


def train_model(celsius, fahrenheit):
    #assemble the layers into the model
    l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

    model = tf.keras.models.Sequential([l0])

    #compile the model with the loss and optimizer
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

    #declaring history to be global
    global history
    
    #then train the model
    history = model.fit(celsius, fahrenheit, epochs=500, verbose=False)

    #output by .predict
    predicts = model.predict([100.0])

    print("Finished training the model")
    return predicts


#instantiate the function
train_model(celsius_q, fahrenheit_a)


#END OF MODEL TRAINING






















