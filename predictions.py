import tensorflow as tf
import math
import logging
import numpy as np
import model_build

# predicting images from an array of test dataset
def predict_images():
    for images, labels in model_build.testing_data_array.take(1):
        images = images.np()
        labels = labels.np()
        
        predictions = model_build.ready_model.predict(images) #predicts each image and its corresponding test-label
        
        # image shape prediction
        image_shapes_predictions = predictions.shape
        
        # single/first prediction => result must be a 10 array format
        image1_prediction = predictions[0]
        
        return predictions, image_shapes_predictions, image1_prediction

prediction_result = predict_images()

# predicting a single image
def single_img_predict():
    image = image[0]
    
    # create an array since tf.keras works with arrays
    # using numpy, our image will be:
    image = np.array(image)
    
    prediction = model_build.testing_data_array.predict(image)
    
    image_shape_prediction = prediction.shape
    
    return prediction, image_shape_prediction

single_image_prediction = single_img_predict()




