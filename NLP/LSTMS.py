import tensorflow as tf
import tensorflow.keras.preprocessing.sequence import pad_sequencies
import pandas
import tensorflow_datasets as tfds
import numpy
import matplotlib.pyplot as plot

# global decralation of variables
vocabularly_size = 100;

max_length = 50
trunc_type='post'
padding_type='post'

embedding_dimension = 16

num_of_epocs = 30

dataset_path = "../Resources/tensorflow_datasets/sentiment.csv"

""" read our dataset
    parameters to pass, path of the dataset
    Extract out sentences and labels
"""
def getLocalDataset(dataset_path):
    dataset = pandas.read_csv(dataset_path)
    
    sentences = dataset['text'].tolist()
    labels = dataset['sentiment'].tolist()
    
    return [sentences, labels]
    
sentences = getLocalDataset(dataset_path)[0]
labels = getLocalDataset(dataset_path)[1]

"""SubwordTextEncoder.build_from_corpus() will create a tokenizer for us.
    Replace sentence data with encoded subwords(encode into subwords)
"""
def createSubwords(sentences, vocabularly_size):
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(sentences, vocabularly_size, max_subword_length = 5)
    
    return tokenizer


tokenizer = createSubwords(sentences, vocabularly_size)

""" we need to pad the sequences, as well as split into training and test sets.
    :-  # Pad all sequences
        # Separate out the sentences and labels into training and test sets
        # Make labels into numpy arrays for use with the network later
"""
def finalDataProcessing(sentencies, labels, max_length, trunc_type, padding_type):
    
    padded_sequencies = pad_sequencies(sentences, maxlen = max_length, padding = padding_type, truncating = trunc_type)
    
    training_size = int(len(sentences) * 8)
    
    training_sequencies = padded_sequencies[0:training_size]
    testing_squencies = padded_sequencies[training_size:]
    
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]
    
    final_training_labels = numpy.array(training_labels)
    final_testing_labels = numpy.array(testing_labels)
    
    return [training_sequencies, testing_squencies, final_training_labels, final_testing_labels]

sorted_datasets = finalDataProcessing(sentences, labels, max_length, trunc_type, padding_type)

"""
    Create the model using an Embedding
    
"""
def createModel(max_length, vocabularly_size, embedding_dimension):
    model = tf.keras.Sequential({
        tf.keras.layers.Embedding(vocabularly_size, embedding_dimension, input_length = max_length),
        # tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dimension, return_sequencies = True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dimension)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    })
    
    return model
model = createModel(max_length, vocabularly_size, embedding_dimension)

"""
    training the model
    :- compile $ fit.
"""
def trainModel(model, num_of_epocs, sorted_datasets):
    
    model = model.compile(loss='binary_crossentrophy', optimizer='adam', metrics = ['accuracy'])
    history = model.fit(sorted_datasets[0], sorted_datasets[2], epochs = num_of_epocs, validation_data = (sorted_datasets[1], sorted_datasets[3]) )
    
    return [model, history]

trained_model = trainModel(model, num_of_epocs, sorted_datasets)

def graphicalRepresentation(history):
    plot.plot(history.history[string])
    plot.plot(history.history['val_'+string])
    plot.xlabel("Epochs")
    plot.ylabel(string)
    plot.legend([string, 'val_'+string])
    plot.show()

graphicalRepresentation(trained_model[1], "accuracy")
graphicalRepresentation(trained_model[1], "loss")


"""
    Define a function to take a series of reviews
    and predict whether each one is a positive or negative review

    max_length = 100 # previously defined
    
    Keep the original sentences so that we can keep using them later
    Create an array to hold the encoded sequences
    Convert the new reviews to sequences
    Pad all sequences for the new reviews
"""
def predict_review(model, new_sentences, max_length, show_padded_sequence=True):
    new_sequences = []

    for i, frvw in enumerate(new_sentences):
        new_sequences.append(tokenizer.encode(frvw))

    trunc_type = 'post'
    padding_type = 'post'

    new_reviews_padded = pad_sequences(new_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    classes = model.predict(new_reviews_padded)

    for x in range(len(new_sentences)):
        
        if (show_padded_sequence):
            print(new_reviews_padded[x])
            
            print(new_sentences[x])
            
            print(classes[x])
            print("\n")

# Use the model to predict some/any reviews available for prediction   
fake_reviews = ["I love this phone", 
                "Everything was cold",
                "Everything was hot exactly as I wanted", 
                "Everything was green", 
                "the host seated us immediately",
                "they gave us free chocolate cake", 
                "we couldn't hear each other talk because of the shouting in the kitchen"
            ]

predict_review(trained_model, fake_reviews, max_length)
