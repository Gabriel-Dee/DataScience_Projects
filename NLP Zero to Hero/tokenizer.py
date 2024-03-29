# Part 1
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
print(word_index)