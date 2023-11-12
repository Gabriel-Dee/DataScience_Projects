import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer # getting the tokenizer APIs
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You Love my dog!',
    'Do you think my dog is amaizing?'
]

tokenizer = Tokenizer(num_words = 100) # creating an instance of a tokenizer object, num_words parameter represents max no of words to keep
tokenizer.fit_on_texts(sentences) # telling the tokenizer to go through the texts and fitting on them like this
word_index = tokenizer.word_index # the full list of words is available as the tokenizers word_index property

sequences = tokenizer.texts_to_sequences(sentences) # creating sequences of texts in a sentence
padded = pad_sequences(sequences, maxlen=5)
# padded = pad_sequences(sequences, maxlen=5, padding='post', truncating='post')

print("\nWord Index = " , word_index)
print("\nSequences = " , sequences)
print("\nPadded Sequences:")
print(padded)


# Try with words that the tokenizer wasn't fit to
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)

padded = pad_sequences(test_seq, maxlen=10)
print("\nPadded Test Sequence: ")
print(padded)