#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: shariq
"""

from keras.models import Model
from keras.layers import Embedding, Dense, Input, LSTM, Bidirectional, Activation
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten
from keras.layers.merge import concatenate
from keras.models import Sequential
from keras.models import load_model
from keras import backend as K

class ModelBuilder:
    
    def __init__(self, vocab_size = None, embedding_matrix = None):
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
    
    def multi_input_model(self, pr_len, comm_len, optimizer = 'adam', loss='binary_crossentropy'):
        with K.name_scope('input_parent'):
            input_parent = Input(shape=(pr_len,), name='input_parent')
            embedding1 = Embedding(self.vocab_size, 100, 
                                   weights=[self.embedding_matrix], 
                                   input_length=pr_len, trainable=False, 
                                   name='embedding_parent')(input_parent)
        
        with K.name_scope('input_comment'):
            input_comment = Input(shape=(comm_len,), name='input_comment')
            embedding2 = Embedding(self.vocab_size, 100, 
                                   weights=[self.embedding_matrix], 
                                   input_length=comm_len, trainable=False, 
                                   name='embedding_comment')(input_comment)
        
        with K.name_scope('main_flow'):
            x = concatenate([embedding2, embedding1], axis=1, name='concat')
            x = Bidirectional(LSTM(20, return_sequences=True, name='bilstim_1'))(x)
            x = Dense(20, activation='sigmoid', name='dense_1')(x)
            x = Bidirectional(LSTM(20, name='bilstm_2'))(x)
            output = Dense(2, activation='sigmoid', name='output')(x)
        
        model = Model(inputs=[input_parent, input_comment], outputs=[output])
        model.compile(loss=loss, optimizer=optimizer, metrics = ['accuracy'])
        print(model.summary())
        return model
    
    def transfer_learning_model(self, model=None, model_path=None):        
        if model_path is not None:
            model = load_model(model_path)
        # remove the output layer, bi-lstm-2 and dense-layer
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        
        for l in model.layers:
            l.trainable = False
        
        x = model.layers[-1].output
        print(x.name)
#        x = Bidirectional(LSTM(20, return_sequences=True, name='tl_bilstm_1'), name='bi_1')(x)
        x = Dense(20, activation='sigmoid', name='tl_dense_1')(x)
        x = Bidirectional(LSTM(20, name='tl_bilstm_2'), name='bi_2')(x)
        output = Dense(2, activation='sigmoid', name='tl_output')(x)
        
        new_model = Model(inputs=model.input, outputs=[output])
        new_model.compile(loss=model.loss, optimizer='adam', metrics = ['accuracy'])
        print(new_model.summary())
        return new_model
    
    def multi_input_conv_model(self, pr_len, comm_len, optimizer = 'adam', loss='binary_crossentropy'):
        with K.name_scope('input_parent'):
            input_parent = Input(shape=(pr_len,), name='input_parent')
            embedding1 = Embedding(self.vocab_size, 100, 
                                   weights=[self.embedding_matrix], 
                                   input_length=pr_len, trainable=False, 
                                   name='embedding_parent')(input_parent)
        
        with K.name_scope('input_comment'):
            input_comment = Input(shape=(comm_len,), name='input_comment')
            embedding2 = Embedding(self.vocab_size, 100, 
                                   weights=[self.embedding_matrix], 
                                   input_length=comm_len, trainable=False, 
                                   name='embedding_comment')(input_comment)
        
        with K.name_scope('main_flow'):
            x = concatenate([embedding2, embedding1], axis=1, name='concat')
            
            conv1 = Conv1D(filters=32, kernel_size=4, activation='sigmoid')(x)
            conv1 = Dropout(0.2)(conv1)
            conv1 = MaxPooling1D(pool_size=2)(conv1)
            
            x = Conv1D(filters=32, kernel_size=6, activation='sigmoid')(x)
            x = Dropout(0.2)(x)
            x = MaxPooling1D(pool_size=2)(x)
            
            x = Conv1D(filters=32, kernel_size=8, activation='sigmoid')(x)
            x = Dropout(0.2)(x)
            x = MaxPooling1D(pool_size=2)(x)
            
            x = Dense(20, activation='sigmoid', name='dense_1')(x)
            x = Bidirectional(LSTM(20, return_sequences=True, name='bilstim_1'))(x)
            x = Dense(20, activation='sigmoid', name='dense_2')(x)
            x = Bidirectional(LSTM(20, name='bilstm_2'))(x)
            output = Dense(2, activation='sigmoid', name='output')(x)
        
        model = Model(inputs=[input_parent, input_comment], outputs=[output])
        model.compile(loss=loss, optimizer=optimizer, metrics = ['accuracy'])
        print(model.summary())
        return model
