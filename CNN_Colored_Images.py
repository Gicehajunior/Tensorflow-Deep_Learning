import tensorflow as tf
import matplotlib.pyplot as plot
import logging
import os
import cv2

#global declaration of batch size
BATCH_SIZE = ""
image_shape = ""

train_directory = 'Resources/microsoft_datasets/cats_and_dogs/tests_cats_and_dogs'
validation_directory = 'Resources/microsoft_datasets/cats_and_dogs/train_cats_and_dogs'
categories = ["cats", "dogs"]
#telling the tensorflow to log the errors if there is
def logging():
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)
    
logging()


def load_images(train_directory, test_directory, data_categories):
    # preprocessing train data=> of cats and dogs
    for category in data_categories:
        train_data_path = os.path.join(train_directory, category)
        test_data_path = os.path.join(test_directory, category)
        
        for train_image in os.listdir(train_data_path):
            # reading the train data for each image
            #and load them in Grayscale format
            
            train_image_array = cv2.imread(os.path.join(train_data_path, train_image))
            
            # you can shjow the image like 
            # plot.imshow(train_image_array, cmap="gray")
            # print(train_image_array)
            # plot.show()
            
        for test_image in os.listdir(test_data_path):
            # reading the train data for each image
            #and load them in Grayscale format
            
            test_image_array = cv2.imread(os.path.join(test_data_path, test_image))
            
            # you can show the image like 
            
            # plot.imshow(test_image_array, cmap="gray")
            # plot.show()
            
        # dataset total per folder
        train_total_data = len(os.listdir(train_data_path))
        test_total_data =len(os.listdir(test_data_path))
        total_dataset = train_total_data + test_total_data  
        
        dataset = [train_image_array, test_image_array, train_total_data, test_total_data, ]

        return dataset


dataset_directories = load_images(
    train_directory,
    validation_directory,
    categories
    )

#Images must be formatted into appropriately pre-processed floating point tensors before being fed into the network. 
# #The steps involved in preparing these images are:

    #Read images from the disk
    #Decode contents of these images and convert it into proper grid format as per their RGB content
    #Convert them into floating point tensors
    #Rescale the tensors from values between 0 and 255 to values between 0 and 1, as neural networks prefer to deal with small input values.
    #preprocess using ImageDataGenerator
def format_pictures(training_data, testing_data):
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
    
    global BATCH_SIZE
    global image_shape
    
    BATCH_SIZE = 100
    image_shape = 150

    test_data_generator = test_image_generator.flow_from_directory(
        batch_size = BATCH_SIZE,
        directory = testing_data,
        shuffle = False,
        target_size = (image_shape, image_shape),
        class_mode = 'binary'
        )
    
    train_data_generator = train_image_generator.flow_from_directory(
        batch_size = BATCH_SIZE,
        directory = training_data,
        shuffle = True,
        target_size = (image_shape, image_shape),
        class_mode = 'binary'
        )
    
    dataset = [train_data_generator, test_data_generator]
    
    return dataset

formated_images = format_pictures(dataset_directories[0], dataset_directories[1])

test_dir_gen = formated_images[0]
train_dir_gen = formated_images[1]

# # CREATE THE MODEL NOW
def model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, 
            (3, 3), 
            padding='same', 
            activation=tf.nn.relu, 
            input_shape=(150,150,3)
            ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    
    return model
model = model()

def optimize(unoptimized_model):
    model = unoptimized_model.compile(
        optimizer='adams',
        loss=tf.keras.losses.SparseCategoricalCrossentrophy(from_logits=True),
        metrics=['accuracy']
        )
    return model

optimized_model = optimize(model)

# we can now train our model
def train_model(model, train_total_data, test_total_data, train_data_dir, test_data_dir):
    # Since our batches are coming from a generator(ImageDataGenerator), we'll use fit_generator instead of fit.
    # call the data and train our model
    model = model.fit_generator(train_data_dir, epochs=100, steps_per_epoch=int(np.ceil(train_total_data / float(BATCH_SIZE))))
    
    
    model = model.fit_generator(test_data_dir, epochs=100, steps_per_epoch=int(np.ceil(test_total_data / float(BATCH_SIZE))))
    return model

train_model(optimized_model, dataset_directories[3], dataset_directories[4], train_dir_gen, test_dir_gen)

# # The model now ready for predictions
