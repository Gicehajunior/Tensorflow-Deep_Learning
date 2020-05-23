import tensorflow as tf
import matplotlib.pyplot as plot
import model_train 
import logging
import os

# global declaration of batch size
BATCH_SIZE = ""
image_shape = ""

model_train.logging()

def load_images_from_folder(train_cat_folder, train_dog_folder, test_cat_folder, test_dog_folder):
    # return images
    # train data
    train_cat_data_dir = os.listdir(train_cat_folder)
    train_dog_data_dir = os.listdir(train_dog_folder)
    
    # test data
    test_cat_data_dir = os.listdir(test_cat_folder)
    test_dog_data_dir = os.listdir(test_dog_folder)
    
    # dataset total per folder
    train_total_data = len(train_cat_data_dir) + len(train_dog_data_dir)
    test_total_data =len(test_cat_data_dir) + len(test_dog_data_dir) 
    total_dataset = train_total_data + test_total_data  
    
    dataset = [train_cat_data_dir, train_dog_data_dir, test_cat_data_dir, test_dog_data_dir, train_total_data, test_total_data]
    
    return dataset


dataset_directories = load_images_from_folder(
    'Resources/microsoft_datasets/cats_and_dogs/train_cats_and_dogs/cats', 
    'Resources/microsoft_datasets/cats_and_dogs/train_cats_and_dogs/dogs',
    'Resources/microsoft_datasets/cats_and_dogs/tests_cats_and_dogs/cata',
    'Resources/microsoft_datasets/cats_and_dogs/tests_cats_and_dogs/dogs'
    )

#Images must be formatted into appropriately pre-processed floating point tensors before being fed into the network. 
# #The steps involved in preparing these images are:

    #Read images from the disk
    #Decode contents of these images and convert it into proper grid format as per their RGB content
    #Convert them into floating point tensors
    #Rescale the tensors from values between 0 and 255 to values between 0 and 1, as neural networks prefer to deal with small input values.
    #preprocess using ImageDataGenerator
def format_pictures(image_dir1, image_dir2):
    train_image_generator = ImageDataGenerator(
        rescale=1./255,
        
        # to get rid of overfitting
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )
    test_image_generator = ImageDataGenerator(
        rescale=1./255,
        
        # to get rid of overfitting
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )
    
    global BATCH_SIZE = 100
    global image_shape = 150
    
    
    test_data_generator = test_image_generator.flow_from_directory(
        batch_size = BATCH_SIZE,
        directory = image_dir1,
        shuffle = False,
        target_size = (image_shape, image_shape),
        class_mode = 'binary'
        )
    
    train_data_generator = train_image_generator.flow_from_directory(
        batch_size = BATCH_SIZE,
        directory = image_dir2,
        shuffle = True,
        target_size = (image_shape, image_shape),
        class_mode = 'binary'
        )
    
    dataset = [test_data_generator, train_data_generator]
    
    return dataset

formated_images = format_pictures(dataset_directories[0], dataset_directories[1])

test_dir_gen = formated_images[0]
train_dir_gen = formated_images[1]

# CREATE THE MODEL NOW
def model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu input_shape=(150,150,3))
        tf.keras.layers.MaxPooling2D(2, 2)
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu)
        tf.keras.layers.MaxPooling2D(2, 2)
        tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu)
        tf.keras.layers.MaxPooling2D(2, 2)
        tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu)
        tf.keras.layers.MaxPooling2D(2, 2)
        tf.keras.layers.flatten()
        tf.keras.layers.Dense(512, activation=tf.nn.relu)
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    
    return model
model = model()

def optimize(unoptimized_model):
    model = unoptimized_model.compile(
        optimizer='adams',
        loss=tf.keras.losses=SparseCategoricalCrossentrophy(from_logits=True),
        metrics['accuracy']
        )
    return model

optimized_model = optimize(model)

# we can now train our model

def train_model(model, train_data_dir, test_data_dir):
    # Since our batches are coming from a generator(ImageDataGenerator), we'll use fit_generator instead of fit.
    # call the data and train our model
    model = model.fit_generator(train_data_dir, epochs=100, steps_per_epoch=int(np.ceil(dataset_directories[4] / float(BATCH_SIZE))))
    
    
    # model = model.fit_generator(test_data_dir, epochs=100, steps_per_epoch=int(np.ceil(dataset_directories[5] / float(BATCH_SIZE))))
    return model

train_model(optimized_model, train_dir_gen, test_dir_gen)
    

# The model now ready for predictions
