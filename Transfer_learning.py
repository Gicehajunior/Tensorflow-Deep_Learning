import tensorflow as tf
import matplotlib.pylab as plot

import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import layers
import os

import logging
import model_train

# global declarationions of variables
imageNet_labels = 'Resources/tensorflow_datasets'
pretrained_model_path = 'Resources/pretrained_models/tensorflow_models/modileNet_V2_tf_model.pb'
train_directory = 'Resources/microsoft_datasets/cats_and_dogs/tests_cats_and_dogs'
validation_directory = 'Resources/microsoft_datasets/cats_and_dogs/train_cats_and_dogs'
categories = ["cats", "dogs"]

model_train.logging()


def load_images(train_directory, test_directory, data_categories):
    # preprocessing train data=> of cats and dogs
    for category in data_categories:
        train_data_path = os.path.join(train_directory, category)
        test_data_path = os.path.join(test_directory, category)

        for train_image in os.listdir(train_data_path):
            # reading the train data for each image
            #and load them in Grayscale format

            train_image_array = cv2.imread(
                os.path.join(train_data_path, train_image))

            # you can shjow the image like
            # plot.imshow(train_image_array, cmap="gray")
            # print(train_image_array)
            # plot.show()

        for test_image in os.listdir(test_data_path):
            # reading the train data for each image
            #and load them in Grayscale format

            test_image_array = cv2.imread(
                os.path.join(test_data_path, test_image))

            # you can show the image like

            # plot.imshow(test_image_array, cmap="gray")
            # plot.show()

        # dataset total per folder
        train_total_data = len(os.listdir(train_data_path))
        test_total_data = len(os.listdir(test_data_path))
        total_dataset = train_total_data + test_total_data

        dataset = [train_image_array, test_image_array, train_total_data, test_total_data]
        
        return dataset


dataset_directories = load_images(
    train_directory,
    validation_directory,
    categories
)

def labels(imageNet_txt_filepath):
    # you can check for errors ie. OS error: OSError by using try and catch
    global labels
    
    labels_path = tf.keras.utils.get_file(imageNet_labels.txt, imageNet_txt_filepath)
    imageNet_labels = np.array(open(labels).read().splitlines())
    
    return imageNet_labels

image_labels = labels(imageNet_directory)

def feature_extractor(pre_model_path):
    pretrained_model = tf.saved_model.load(pretrained_model)
    
    # using our loaded model to extract features on our images
    feature_extractor = hub.KerasLayer(pretrained_model, input_shape=(IMAGE_RES, IMAGE_RES,3))
    
    # freeze the variables in this extractor layer
    feature_extractor.trainable = false
    
    return feature_extractor

ready_feature_extractor = feature_extractor(pretrained_model_path)

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
        metrics=['accuracy']
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
# formats the images into required sizes ie. (24, 24)
def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0

    return image, label

train_batches = dataset_directories[0].cache().shuffle(dataset_directories[2]//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

test_batches = dataset_directories[1].cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

model = train_model(train_batches, test_batches)
# Remember our model object is still the full MobileNet model trained on ImageNet, 
# so it has 1000 possible output classes. 
# ImageNet has a lot of dogs and cats in it, 
# so let's see if it can predict the images in our Dogs vs. Cats dataset
# creating mobileNet in Keras =>https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py 
def predict_images(labels, image_batches, model):
    class_names = np.array(labels)
    predicted_image_batch = model.predict(image_batches)
    predicted_image_batch = tf.squeeze(predicted_image_batch).np()
    predicted_image_ids = np.argmax(predicted_image_batch, axis=-1)
    
    predicted_class_names = class_names[predicted_image_ids]
    
    return predicted_class_names

predicts = predict_images(image_labels, train_batches, model, predicted_image_ids)

def plot_images(image_batch, predicted_ids, label_batch, classnames_predicted, classnames_predicted):
    plt.figure(figsize=(10,9))
    
    for n in range(30):
        plt.subplot(6,5,n+1)
        plt.subplots_adjust(hspace = 0.3)
        plt.imshow(image_batch[n])
        color = "blue" if predicted_ids[n] == label_batch[n] else "red"
        plt.title(classnames_predicted[n].title(), color=color)
        plt.axis('off')
        _= plt.suptitle("Model predictions (blue: correct, red: incorrect)")

plot_images(predicts[0], train_batches, predicts[2], predicts[3], predicted_class_names)




