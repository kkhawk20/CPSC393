"""
# Homework 4 (Sequential Models)

1. Choose a book or other text to train your model on (I suggest [Project Gutenberg](https://www.gutenberg.org/ebooks/) to find .txt files but you can find them elsewhere). Make sure your file is a `.txt` file. Clean your data (You may use [this file](https://colab.research.google.com/drive/1HCgKn5XQ7Q3ywxGszVWx2kddfT9UBASp?usp=sharing) we talked about in class as a baseline). Build a sequential model (LSTM, GRU, SimpleRNN, or Transformer) that generates new lines based on the training data (NO TRANSFER LEARNING). While your model doesn't need to have perfect accuracy, you must have an appropriate architecture and train it for a reasonable amount of epochs.

Print out or write 10 generated sequences from your model (Similar to Classwork 17 where we generated new Pride and Prejudice lines, but now with words instead of charachters. Feel free to use [this]() as a reference for how to generate text from a trained model). Assess in detail how good they are, what they're good at, what they struggle to do well. INCLUDE THESE 10 sequences in your report.

2. Make a new model with ONE substantial adjustment (e.g. use a custom embedding layer if you didn't already, use a pre-trained embedding layer if you didn't already, use a DEEP LSTM/GRU with multiple recurrent layers, use a pre-trained model to do transfer learning and fine-tune it...etc.). While your model doesn't need to have perfect accuracy, you must have an appropriate architecture and train it for a reasonable amount of epochs.

Print out or write 10 generated sequences from your model (Similar to Classwork 17 where we generated new Pride and Prejudice lines, but now with words instead of charachters. Feel free to use [this]() as a reference for how to generate text from a trained model. INCLUDE THESE 10 sequences in your report. Assess in detail how good they are, what they're good at, what they struggle to do well.  Did the performance of your model change?

3. Then create a **technical report** discussing your model building process, the results, and your reflection on it. The report should follow the format in the example including an Introduction, Analysis, Methods, Results, and Reflection section.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.text import Tokenizer
import string
from random import randint
from pickle import load
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

book_dir = '/app/rundir/CPSC393/HW4/book.txt'
save_dir = '/app/rundir/CPSC393/HW4/'

"""# Loading data and cleaning/setup"""

# Read in the file
data_source = open(book_dir, 'r', encoding='utf-8').read()
data = data_source.lower()

# create mapping of unique words to integers
words = data.split()
unique_words = sorted(set(words))
word_to_int = {word: i for i, word in enumerate(unique_words)}

n_words = len(words)  # Total number of words in the text.
n_vocab = len(unique_words)  # Number of unique words (vocabulary size).
# print("Total Words: ", n_words)
# print("Total Vocab: ", n_vocab)

# changeable params
my_file = book_dir
seq_len = 100

# load doc into memory
def load_doc(filename):
 # open the file as read only
 file = open(filename, 'r')
 # read all text
 text = file.read()
 # close the file
 file.close()
 return text

# turn a doc into clean tokens
def clean_doc(doc):
 # replace '--' with a space ' '
 doc = doc.replace('--', ' ')
 # split into tokens by white space
 tokens = doc.split()
 # remove punctuation from each token
 table = str.maketrans('', '', string.punctuation)
 tokens = [w.translate(table) for w in tokens]
 # remove remaining tokens that are not alphabetic
 tokens = [word for word in tokens if word.isalpha()]
 # make lower case
 tokens = [word.lower() for word in tokens]
 return tokens

# save tokens to file, one dialog per line
def save_doc(lines, filename):
 data = '\n'.join(lines)
 file = open(filename, 'w')
 file.write(data)
 file.close()

# load document
doc = load_doc(my_file)
# print(doc[:50])

# clean document
tokens = clean_doc(doc)

with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
    f.write('Total Tokens: %d' % len(tokens) + "\n")
    f.write('Unique Tokens: %d' % len(set(tokens)) + "\n")


# organize into sequences of tokens
length = seq_len + 1
sequences = list()
for i in range(length, len(tokens)):
 # select sequence of tokens
 seq = tokens[i-length:i]
 # convert into a line
 line = ' '.join(seq)
 # store
 sequences.append(line)
# print('Total Sequences: %d' % len(sequences))
 
with open(os.path.join(save_dir, "metrics.txt"), "a") as f:
    f.write('Total Sequences: %d' % len(sequences) + "\n")

# save sequences to file
out_filename = my_file[:-4] + '_seq.txt'
save_doc(sequences, out_filename)

# load
doc = load_doc(out_filename)
lines = doc.split('\n')

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

# vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# separate into input and output
sequences = np.array(sequences)
sequences.shape
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

p_train = 0.8

n_train = int(X.shape[0]//(1/p_train))
X_train = X[0:n_train]
y_train = y[0:n_train]
X_test = X[n_train:]
y_test = y[n_train:]




"""# 1. LSTM - basic"""

# LSTM model

vocab_size = len(word_to_int) + 1
embedding_dim = 50  # Size of the embedding vectors
max_length = 100  # Sequence length, each input sequence is 100 words long

inputs = Input(shape=(max_length,))
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(inputs)
x = LSTM(100)(embedding)
x = Dropout(0.2)(x)
output = Dense(y.shape[1], activation='softmax')(x)

model = Model(inputs = inputs, outputs = output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    mode = 'min',
    restore_best_weights=True,
    verbose = 1)

model_checkpoint = ModelCheckpoint(
    filepath='./best_model.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1)

history = model.fit(X_train, y_train, epochs=500, batch_size=128,
                    validation_data=(X_test, y_test),
                    callbacks = [early_stopping, model_checkpoint])

# Accuracy/val_accuracy
# print("Accuracy: ", history.history['accuracy'][:1])
# print("Val Accuracy: ",history.history['val_accuracy'][:1])

# print("Loss: ", history.history['loss'][:1])
# print("Val Loss: ", history.history['val_loss'][:1])

# Printing those metrics to a file
with open(os.path.join(save_dir, "metrics.txt"), "a") as f:
    f.write("Basic Model Accuracy: " + str(history.history['accuracy'][:1]) + "\n")
    f.write("Basic Model Val Accuracy: " + str(history.history['val_accuracy'][:1]) + "\n")
    f.write("Basic Model Loss: " + str(history.history['loss'][:1]) + "\n")
    f.write("Basic Model Val Loss: " + str(history.history['val_loss'][:1]) + "\n")

# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(save_dir, "accuracy_plot.png"))

# Plot training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(save_dir, "loss_plot.png"))

# load doc into memory
def load_doc(filename):
 # open the file as read only
 file = open(filename, 'r')
 # read all text
 text = file.read()
 # close the file
 file.close()
 return text
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
  result = []
  in_text = seed_text
  for _ in range(n_words):
    # Encode the text as integer
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    # Truncate sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    probabilities = model.predict(encoded, verbose=0).flatten()
    probabilities = np.exp(probabilities / .8)  # Applying temperature scaling
    probabilities /= np.sum(probabilities)
    yhat = np.random.choice(range(len(probabilities)), p=probabilities)
    # Ensure yhat is treated as an integer for indexing
    out_word = tokenizer.index_word[yhat] if yhat.size > 0 else ''
    # Append to input for generating the next word
    in_text += ' ' + out_word
    result.append(out_word)
  return ' '.join(result)

# load cleaned text sequences
in_filename = "/app/rundir/CPSC393/HW4/book_seq.txt"
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

# select a seed text
seed_text = lines[randint(0,len(lines))]
# print(seed_text + '\n')

# generate new text
best_model = load_model('./best_model.h5')
generated = generate_seq(best_model, tokenizer, seq_length, seed_text, 50)
# print(generated)

# Printing to file
with open(os.path.join(save_dir, "metrics.txt"), "a") as f:
    f.write("Seed Text: " + seed_text + "\n")
    f.write("Generated Text: " + generated + "\n")

"""# 2. LSTM with DEEPER learning - More Layers"""

# LSTM model DEEPER!!
from tensorflow.keras.regularizers import l1, l2

vocab_size = len(word_to_int) + 1
embedding_dim = 50  # Size of the embedding vectors
max_length = 100  # Sequence length, each input sequence is 100 words long

inputs = Input(shape=(max_length,))
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(inputs)
x = LSTM(256, return_sequences=True,  # Set to False if this is the last LSTM layer before the Dense output layer
                  kernel_regularizer=l2(0.01),  # Applies L2 regularization to the input weights
                  recurrent_regularizer=l1(0.01),  # Applies L1 regularization to the recurrent weights
                  bias_regularizer=l2(0.01)  # Applies L2 regularization to the bias
                  )(embedding)
x = Dropout(0.2)(x)
x = LSTM(256, return_sequences=True,  # Set to False if this is the last LSTM layer before the Dense output layer
                  kernel_regularizer=l2(0.01),  # Applies L2 regularization to the input weights
                  recurrent_regularizer=l1(0.01),  # Applies L1 regularization to the recurrent weights
                  bias_regularizer=l2(0.01)  # Applies L2 regularization to the bias
                  )(x)
x = Dropout(0.2)(x)
x = LSTM(256, return_sequences=False,  # Set to False if this is the last LSTM layer before the Dense output layer
                  kernel_regularizer=l2(0.01),  # Applies L2 regularization to the input weights
                  recurrent_regularizer=l1(0.01),  # Applies L1 regularization to the recurrent weights
                  bias_regularizer=l2(0.01)  # Applies L2 regularization to the bias
                  )(x)
x = Dropout(0.2)(x)
x = Dense(256, activation = 'relu')(x)
output = Dense(y.shape[1], activation='softmax')(x)

model = Model(inputs = inputs, outputs = output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

def scheduler(epoch, lr):
    if epoch < 10:
        return lr  # No change in learning rate for the first 10 epochs
    else:
        return lr * 0.99  # Reduce the learning rate by 10% every epoch after the 10th

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    mode = 'min',
    verbose = 1)

lr_scheduler = LearningRateScheduler(scheduler)

model_checkpoint = ModelCheckpoint(
    filepath='./best_model_2.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1)

history = model.fit(X_train, y_train, epochs=500, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks = [early_stopping, model_checkpoint])

# Accuracy/val_accuracy
# print("Accuracy: ", history.history['accuracy'][:1])
# print("Val Accuracy: ",history.history['val_accuracy'][:1])

# print("Loss: ", history.history['loss'][:1])
# print("Val Loss: ",history.history['val_loss'][:1])

# Printing those metrics to a file
with open(os.path.join(save_dir, "metrics.txt"), "a") as f:
    f.write("Deep Model Accuracy: " + str(history.history['accuracy'][:1]) + "\n")
    f.write("Deep Model Val Accuracy: " + str(history.history['val_accuracy'][:1]) + "\n")
    f.write("Deep Model Loss: " + str(history.history['loss'][:1]) + "\n")
    f.write("Deep Model Val Loss: " + str(history.history['val_loss'][:1]) + "\n")

# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(save_dir, "accuracy_ploD.png"))

# Plot training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(save_dir, "loss_ploD.png"))


# load doc into memory
def load_doc(filename):
 # open the file as read only
 file = open(filename, 'r')
 # read all text
 text = file.read()
 # close the file
 file.close()
 return text
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
  result = []
  in_text = seed_text
  for _ in range(n_words):
    # Encode the text as integer
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    # Truncate sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    probabilities = model.predict(encoded, verbose=0).flatten()
    probabilities = np.exp(probabilities / .8)  # Applying temperature scaling
    probabilities /= np.sum(probabilities)
    yhat = np.random.choice(range(len(probabilities)), p=probabilities)
    # Ensure yhat is treated as an integer for indexing
    out_word = tokenizer.index_word[yhat] if yhat.size > 0 else ''
    # Append to input for generating the next word
    in_text += ' ' + out_word
    result.append(out_word)
  return ' '.join(result)

# load cleaned text sequences
in_filename = "/app/rundir/CPSC393/HW4/book_seq.txt"
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

# select a seed text
seed_text = lines[randint(0,len(lines))]
# print(seed_text + '\n')

# generate new text
best_model = load_model('./best_model_2.h5')
generated = generate_seq(best_model, tokenizer, seq_length, seed_text, 50)
# print(generated)

# Printing to file
with open(os.path.join(save_dir, "metrics.txt"), "a") as f:
    f.write("Seed Text: " + seed_text + "\n")
    f.write("Generated Text: " + generated + "\n")






'''

# '------------------------------------------------'

"""# For fun - a fully tuned LSTM Deep model

* Created by ChatGPT
"""

import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.optimizers import Adam

# Function to build the model, with hyperparameters to tune
def build_model(hp):

  vocab_size = len(tokenizer.word_index) + 1
  embedding_dim = 50  # Size of the embedding vectors
  max_length = 100  # Sequence length, each input sequence is 10 words long

  model = Sequential()
  model.add(Embedding(input_dim=vocab_size, output_dim=hp.Int('embedding_dim', min_value=32, max_value=512, step=32), input_length=max_length))

  for i in range(hp.Int('num_lstm_layers', 1, 3)):
      model.add(LSTM(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                      return_sequences=i < hp.get('num_lstm_layers') - 1))
      model.add(Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))

  model.add(Dense(vocab_size, activation='softmax'))
  model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model

# Create a tuner object
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    directory='my_dir',
    project_name='lstm_hyperparam_tuning'
)

# Early stopping callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Start the hyperparameter search
tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model = tuner.get_best_models(num_models=1)[0]

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Function to generate sequences
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
  result = []
  in_text = seed_text
  for _ in range(n_words):
    # Encode the text as integer
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    # Truncate sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    probabilities = model.predict(encoded, verbose=0).flatten()
    probabilities = np.exp(probabilities / .8)  # Applying temperature scaling
    probabilities /= np.sum(probabilities)
    yhat = np.random.choice(range(len(probabilities)), p=probabilities)
    # Ensure yhat is treated as an integer for indexing
    out_word = tokenizer.index_word[yhat] if yhat.size > 0 else ''
    # Append to input for generating the next word
    in_text += ' ' + out_word
    result.append(out_word)
  return ' '.join(result)

# Select a seed text
seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n')
seq_length = 100  # The sequence length that the model was trained on - replace if different

# Generate new text
generated = generate_seq(best_model, tokenizer, seq_length, seed_text, 50)
print(generated)
'''