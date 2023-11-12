import json

with open('Sarcasm_Headlines_Dataset.json', 'r') as f:
    datastore = json.load(f)

# initiate or create lists
sentences = []
labels = []
urls = []

# load values into the lists one for sentences, one for labels and one for urls
for item in datastore: 
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])


# text pre processing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer # getting the tokenizer APIs
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token = "<oov>")
tokenizer.fit_on_texts(sentences) 
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences) # turn sentences into sequences of tokens
padded = pad_sequences(sequences, padding='post')

print(padded[0]) # pronting the first tokenized sentence
print(padded.shape) # print the shape of the token

# slicing the sequences into training and testing
training_size = 20,000
testing_size = len(datastore)-training_size[1]-1

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_lables = labels[training_size:]

# Ensuring the neural net only sees the training data and not the testing data
vocab_size =23,000
oov_tok = "<oov>"
max_length = 10
padding_type = 'post'
trunc_type = 'post'

tokenizer = Tokenizer(num_words=vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences) # turn sentences into sequences of tokens
training_padded = pad_sequences(training_sequences, maxlen= max_length, padding=padding_type, truncating = trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences) # turn sentences into sequences of tokens
testing_padded = pad_sequences(testing_sequences, maxlen= max_length, padding=padding_type, truncating = trunc_type)

# neural network code
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length), # an embedding where direction of word will be learnt epoch ny epoch
    tf.keras.layers.GlobalAveragePooling1D(), # pooling by adding up the vectors for each word in a sentence
    # then it is fed into the neural network
    tf.keras.layers.Dense(24, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_lables), verbose=2)

sentence = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))