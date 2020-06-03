from tensorflow.keras.preprocessing.text import Tokenizer

sentences =[
    'My favorite food is ice cream',
    'do you like ice cream too?',
    'My dog likes ice cream!',
    "your favorite flavor of icecream is chocolate",
    "chocolate isn't good for dogs",
    "your dog, your cat, and your parrot prefer broccoli"
]


def tokenize_text(texts):
    tokenizer = Tokenizer(num_of_words=100, oov='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    return tokenizer

tokenizer = tokenize_text(sentences)

# to examine the word index on the tokenizer
word_index = tokenizer.word_index

print(word_index)



"""After you tokenize the words, 
the word index contains a unique number for each word. 
However, the numbers in the word index are not ordered. 
Words in a sentence have an order. 
So after tokenizing the words, the next step is to generate sequences for the sentences."""
def generate_text_sequences(context_sentencies):
    sequencies = tokenizer.texts_to_sequences(context_sentences)
    
    return sequences

sequences = generate_text_sequencies(sentences)
print(sequences)














