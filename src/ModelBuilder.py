#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: shariq
"""

from keras.models import Model
from keras.layers import Embedding, Dense, Input, LSTM, Bidirectional, Activation
from keras.layers.merge import concatenate
from keras.models import Sequential

class ModelBuilder:
    
    def __init__(self, vocab_size, embedding_matrix = None):
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        
    def base_model(self, input_length, optimizer = 'adam'):
        model = Sequential()
        model.add(Embedding(self.vocab_size, 100, weights=[self.embedding_matrix], 
                            input_length=input_length, trainable=False))
        model.add(Bidirectional(LSTM(20, return_sequences=True)))
        model.add(Dense(20, activation='sigmoid'))
        model.add(Bidirectional(LSTM(20)))
        model.add(Dense(2))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
        print(model.summary())
        return model 
    
    def multi_input_model(self, pr_len, comm_len, optimizer = 'adam'):
        input_parent = Input(shape=(pr_len,))
        embedding1 = Embedding(self.vocab_size, 100, weights=[self.embedding_matrix], 
                               input_length=pr_len, trainable=False)(input_parent)
    #    drop1 = Dropout(0.3)(embedding1)
    #    bilstm1 = Bidirectional(LSTM(10, return_sequences=True))(embedding1)
    #    droput = Dropout(0.5)(bilstm1)
    #    dense1 = Dense(10, activation='sigmoid')(bilstm1)
        
        input_comment = Input(shape=(comm_len,))
        embedding2 = Embedding(self.vocab_size, 100, weights=[self.embedding_matrix], 
                               input_length=comm_len, trainable=False)(input_comment)
    #    drop2 = Dropout(0.1)(embedding2)
    #    bilstm2 = Bidirectional(LSTM(10, return_sequences=True))(embedding2)
    #    dense2 = Dense(10, activation='sigmoid')(bilstm2)
    
    #    lstm = Bidirectional(LSTM(20, return_sequences=True))
    #    en_pr = lstm(drop1)
    #    en_cm = lstm(drop2)
            
        x = concatenate([embedding2, embedding1], axis=1)
        x = Bidirectional(LSTM(20, return_sequences=True))(x)
        x = Dense(20, activation='sigmoid')(x)
        x = Bidirectional(LSTM(20))(x)
        output = Dense(2, activation='sigmoid')(x)
        
        model = Model(inputs=[input_parent, input_comment], outputs=[output])
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
        print(model.summary())
        return model