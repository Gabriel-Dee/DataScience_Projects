# part 2
# After handling encounter with words not in the word index or corpus
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer # getting the tokenizer APIs

sentences = [
    'I love my dog',
    'I love my cat',
    'You Love my dog!'
]

tokenizer = Tokenizer(num_words = 100) # creating an instance of a tokenizer object, num_words parameter represents max no of words to keep
tokenizer.fit_on_texts(sentences) # telling the tokenizer to go through the texts and fitting on them like this
word_index = tokenizer.word_index # the full list of words is available as the tokenizers word_index property

sequences = tokenizer.texts_to_sequences(sentences) # creating sequences of texts in a sentence

print(word_index)
print(sequences)


# Try with words that the tokenizer wasn't fit to
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)