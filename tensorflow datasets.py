import tensorflow as tf

# pip install -U tensorflow_datasets
import tensorflow_datasets as tsfd
import math
import logging
import numpy as np
import matplotlib.pyplot as plot

tsfd.disable_progress_bar()

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Asset declaration and initialization
dataset, metadata = tsfd.load(
        'fashion_mnist', 
        as_supervised=True, 
        with_info=True
    )
#all clothings class names
def class_names():
    class_names = [
        'T-shirt/top', 
        'Trouser', 
        'Pullover', 
        'Dress', 
        'Coat',
        'Sandal',      
        'Shirt',   
        'Sneaker',  
        'Bag',   
        'Ankle boot'
    ]
    return class_names

cloth_classes = class_names()
train_dataset, test_dataset = dataset['train'], dataset['test']

# explore the dataset
def explore(metadata):
    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples

    print("Number of training examples: {}".format(num_train_examples))
    print("Number of test examples:     {}".format(num_test_examples))

explore(metadata)

def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)

def cache_images(train_data, test_data):
    # The first time you use the dataset, the images will be loaded from disk
    # Caching will keep them in memory, making training faster
    training_data =  train_dataset.cache()
    testing_dataset  =  test_dataset.cache()
    
    return testing_dataset

test_data = cache_images(train_dataset, test_dataset)

# select one image from the set.
def select_each_image():
    # Plotting the image from the test dataset
    for image, label in test_data.take(1):
        break

    # reshape the image
    image = image.numpy().reshape((28,28))

    return image

# Plot the image taken from the set
def plot_1_image(reshaped_image,):
    plot.figure()
    plot.imshow(reshaped_image, cmap=plot.cm.binary)
    plot.colorbar()
    plot.grid(False)
    plot.show()
    
drawnImage = plot_1_image(select_each_image())

# select 25 images from the set
def select_25_images(sets_of_images, cloths_classes):
    plot.figure(figsize=(10,10))
    i = 0
    for (image, label) in sets_of_images.take(25):
        #for each image reshape()
        image = image.numpy().reshape((28,28))
        plot.subplot(5, 5, i+1)
        plot.xticks([])
        plot.yticks([])
        plot.grid(False)
        plot.imshow(image, cmap=plot.cm.binary)
        plot.xlabel(cloths_classes[label])

        #iterate
        i += 1
        
        plot.show()
newDrawnImage = select_25_images(test_data, cloth_classes)

