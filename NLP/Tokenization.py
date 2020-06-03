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



"""Later, when you feed the sequences into a neural network to train a model, the sequences all need to be uniform in size. Currently the sequences have varied lengths, so the next step is to make them all be the same size, either by padding them with zeros and/or truncating them.

Use f.keras.preprocessing.sequence.pad_sequences to add zeros to the sequences to make them all be the same length. By default, the padding goes at the start of the sequences, but you can specify to pad at the end.

You can optionally specify the maximum length to pad the sequences to. Sequences that are longer than the specified max length will be truncated. By default, sequences are truncated from the beginning of the sequence, but you can specify to truncate from the end.

If you don't provide the max length, then the sequences are padded to match the length of the longest sentence.

For all the options when padding and truncating sequences, see"""
def padding_sequences(text_sequences):
    padded = padding_sequences(text_sequences)
    
    return padded


print("\nWord Index = ", word_index)
print("\nSequences = ", sequences)
print("\nPadded Sequences:")
print(padding_sequences(sequences))


# Specify a max length for the padded sequences
padded = pad_sequences(sequences, maxlen=15)
print(padded)

# Put the padding at the end of the sequences
padded = pad_sequences(sequences, maxlen=15, padding="post")
print(padded)

# Limit the length of the sequences, you will see some sequences get truncated
padded = pad_sequences(sequences, maxlen=3)
print(padded)

# Try turning sentences that contain words that
# aren't in the word index into sequences.
# Add your own sentences to the test_data
test_data = [
    "my best friend's favorite ice cream flavor is strawberry",
    "my dog's best friend is a manatee"
]
print(test_data)


"""Here's where the "out of vocabulary" token is used. 
Try generating sequences for some sentences that have words that are not in the word index."""

# Remind ourselves which number corresponds to the
# out of vocabulary token in the word index
print("<OOV> has the number", word_index['<OOV>'], "in the word index.")

# Convert the test sentences to sequences
test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)

# Pad the new sequences
padded = pad_sequences(test_seq, maxlen=10)
print("\nPadded Test Sequence: ")

# Notice that "1" appears in the sequence wherever there's a word
# that's not in the word index
print(padded)











