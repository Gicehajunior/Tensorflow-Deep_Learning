import tensorflow as tf
import matplotlib.pylab as plot

import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import layers
import os

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# global declarationions of variables
image_shape = ""
directory = 'Resources/pretrained_models/tensorflow_models/modileNet_V2_tf_model'
imageNet_directory = 'Resources/pretrained_models/tensorflow_models/imageNet.txt'
labels = ""
BATCH_SIZE = 32


# cats and dogs images
def load_images_from_folder(train_cat_folder, train_dog_folder, test_cat_folder, test_dog_folder):
    # return images
    # train data
    train_cat_data_dir = os.listdir(train_cat_folder)
    train_dog_data_dir = os.listdir(train_dog_folder)
    
    # test data
    test_cat_data_dir = os.listdir(test_cat_folder)
    test_dog_data_dir = os.listdir(test_dog_folder)
    
    # dataset total per folder
    train_total_cat_data = len(train_cat_data_dir)
    train_total_data = len(train_cat_data_dir) + len(train_dog_data_dir)
    test_total_data =len(test_cat_data_dir) + len(test_dog_data_dir) 
    total_dataset = train_total_data + test_total_data  
    
    dataset = [train_cat_data_dir, train_dog_data_dir, test_cat_data_dir, test_dog_data_dir, train_total_data, test_total_data, train_total_cat_data]
    
    return dataset


dataset_directories = load_images_from_folder(
    'Resources/microsoft_datasets/cats_and_dogs/train_cats_and_dogs/cats', 
    'Resources/microsoft_datasets/cats_and_dogs/train_cats_and_dogs/dogs',
    'Resources/microsoft_datasets/cats_and_dogs/tests_cats_and_dogs/cata',
    'Resources/microsoft_datasets/cats_and_dogs/tests_cats_and_dogs/dogs'
    )

# formats the images into required sizes ie. (24, 24)
def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label


def labels(imageNet_txt_filepath):
    # you can checkpr for errors ie. OS error: OSError
    global labels
    
    labels_path = tf.keras.utils.get_file(imageNet.txt, imageNet_txt_filepath)
    imageNet_labels = np.array(open(labels).read().splitlines())
    
    return imageNet_labels

image_labels = labels(imageNet_directory)

def feature_extractor(pre_model_path):
    
    feature_extractor = hub.KerasLayer(os.listdir(pre_model_path), input_shape=(IMAGE_RES, IMAGE_RES,3))
    # freeze the variables in this extractor layer
    feature_extractor.trainable = false
    return feature_extractor

ready_feature_extractor = feature_extractor(directory)

# using a tensorflow hub mobileNet for preditions
def model(feature_extractor): 
    global image_shape
    image_shape = 224
    model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.Layers.Dense(2)
    ])
    
    return model

uncompiled_model = model(ready_feature_extractor)

def compile_model(model):
    
    compiled_model = model.compile(
        optimizer='adams', 
        losses=SparseCategoricalCrossentrophy(from_logits=True), 
        metrics['accuracy']
        )
    
    return compile_model

compiled_model = compile_model(uncompiled_model)
    
def train_model(training_dataset, validation_batches):
    epochs = 6
    
    trained_model = compile_model.fit(
        training_dataset, 
        epochs=epochs, 
        validation_data=validation_batches
        )
    

# resize the cats and dogs images ie. (224, 224)
# remember to cache the batches for easier and fast prediction
train_batches = dataset_directories[0].cache().shuffle(dataset_directories[6]//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

test_batches = dataset_directories[3].cache().map(format_image).batch(BATCH_SIZE).prefetch(1)
model = train_model(train_batches, test_batches)

# Remember our model object is still the full MobileNet model trained on ImageNet, 
# so it has 1000 possible output classes. 
# ImageNet has a lot of dogs and cats in it, 
# so let's see if it can predict the images in our Dogs vs. Cats dataset
# creating mobileNet in Keras =>https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py 
def predict_images(labels, image_batches model):
    class_names = np.array(labels)
    predicted_image_batch = model.predict(train_batches)
    predicted_image_batch = tf.squeeze(predicted_image_batch).np()
    predicted_image_ids = np.argmax(predicted_image_batch, axis=-1)
    predicted_class_names = class_names[predicted_image_ids]
    
    return predicted_class_names

predicts = predict_images(image_labels, train_batches, model)

def plot_images(image_batch, classnames_predicted):
    plt.figure(figsize=(10,9))
    
    for n in range(30):
        plt.subplot(6,5,n+1)
        plt.subplots_adjust(hspace = 0.3)
        plt.imshow(image_batch[n])
        color = "blue" if predicted_ids[n] == label_batch[n] else "red"
        plt.title(classnames_predicted[n].title(), color=color)
        plt.axis('off')
        _ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")

plot_images(train_batches, predicted_class_names)




