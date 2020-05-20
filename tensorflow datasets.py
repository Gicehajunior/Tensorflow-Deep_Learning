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

dataset, metadata = tsdf.load(
        'fashion_mnist', 
        as_supervised=True, 
        with_info=True
    )

train_dataset, test_dataset = dataset['train'], dataset['test']

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()


# Plotting the image from the test dataset
for image, label in test_dataset.take(1):
    break
image = image.numpy().reshape((28,28))

# Plot the image - voila a piece of fashion clothing
plot.figure()
plot.imshow(image, cmap=plot.cm.binary)
plot.colorbar()
plot.grid(False)
plot.show()
