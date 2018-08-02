#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: shariq
"""

from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def cleanup_str(docs):
	# split into tokens by white space
	tokens = docs.split()
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	tokens = ' '.join(tokens)
	return tokens

def init_tokenizer(docs):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(docs)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size

def encode_docs(tokenizer, data):
    # tokenize the texts 
    encoded_comments = tokenizer.texts_to_sequences(data["comment"]) 
    encoded_parent_comments = tokenizer.texts_to_sequences(data["parent_comment"]) 
    # find the max length of comments and parent comments
    comments_len = len(max(encoded_comments, key=len))
    parent_comments_len = len(max(encoded_parent_comments, key=len))
    max_length = comments_len if comments_len > parent_comments_len else parent_comments_len
    # pad the comments and parent comments using the max_length
    padded_comments = pad_sequences(encoded_comments, maxlen=max_length, padding='post')
    padded_parent_comments = pad_sequences(encoded_parent_comments, maxlen=max_length, padding='post')
    return padded_comments, padded_parent_comments

def encode_docs1(tokenizer, data, column):
    # tokenize the texts 
    encoded_comments = tokenizer.texts_to_sequences(data[column]) 
    # find the max length of comments and parent comments
    max_len = len(max(encoded_comments, key=len))
    # pad the comments and parent comments using the max_length
    padded_comments = pad_sequences(encoded_comments, maxlen=max_len, padding='post')
    return max_len, padded_comments
 
def init_glove_embedding():    
    embeddings_index = dict()
    f = open('/home/shariq/MSc/Research/glove.6B/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index
 
def create_embeddings(vocab_size, tokenizer):   
    embeddings_index = init_glove_embedding()
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

