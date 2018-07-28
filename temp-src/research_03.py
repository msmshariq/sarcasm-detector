#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: shariq
"""

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten, Dropout, Input, Bidirectional, Activation, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split


input_politics = pd.read_csv("/home/shariq/MSc/Research/dataset/train-politics.csv", sep='\t',
                    names = ["label", "comment", "parent_comment"])      

# labels
y = np.array(input_politics["label"])

def init_tokenizer(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    return tokenizer

def process_input(tokenizer, text):
    encoded_texts = tokenizer.texts_to_sequences(text)
    max_length = len(max(encoded_texts, key=len))
    processed_texts = pad_sequences(encoded_texts, maxlen=max_length, padding='post')
    return processed_texts

def get_embedding_index():
    embeddings_index = dict()
    file = open('/home/shariq/MSc/Research/glove.6B/glove.6B.100d.txt')
    for line in file:
        	values = line.split()
        	word = values[0]
        	coefs = np.asarray(values[1:], dtype='float32')
        	embeddings_index[word] = coefs
    file.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index

def create_embedding_matrix(vocab_size, tokenizer, embeddings_index):
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
		   embedding_matrix[i] = embedding_vector
           
           



model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                        input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
    