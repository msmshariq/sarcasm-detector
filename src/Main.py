#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:42:52 2018

@author: shariq
"""
import numpy as np
np.random.seed(2018)

from tensorflow import set_random_seed
set_random_seed(2018)

import random as rn
rn.seed(2018) 


import TopicExtractor
import Utils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Embedding, Dense, Input, LSTM, Bidirectional, Activation
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, TFOptimizer
from keras.models import Sequential
import tensorflow as tf

def load_raw_dataset(file_path):
    all_columns = ["label", "comment", "auth", "subreddit", "score", "ups", 
                   "downs", "date", "created_utc", "parent_comment"]
    return pd.read_csv(file_path, sep='\t',names = all_columns)     
        

def load_filtered_dataset(file_path):
    return pd.read_csv(file_path, sep='\t', names = ["label", "comment", "parent_comment"]) 
    

def get_politics_data():
    politics_filter = {'types' : "DBpedia:Politician," +
                   "DBpedia:PoliticalParty," +
                   "DBpedia:OfficeHolder," +
                   "DBpedia:PoliticalParty," +
                   "DBpedia:GovernmentAgency"}

    col_names=["label", "comment", "parent_comment"]
    politcs_extract = TopicExtractor(filter=politics_filter, columns=col_names)
    return politcs_extract
    
#####    


#def base_model(vocb_size, embedding_matrix, length):
def base_model():
    model = Sequential()
    model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len, trainable=False))
    model.add(Bidirectional(LSTM(20, return_sequences=True)))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Bidirectional(LSTM(20)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
#    print(model.summary())
    return model 

def multi_input_model(pr_len, comm_len):
    input_parent = Input(shape=(pr_len,))
    embedding1 = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=pr_len, trainable=False)(input_parent)
    bilstm1 = Bidirectional(LSTM(10, return_sequences=True))(embedding1)
#    dense1 = Dense(10, activation='sigmoid')(bilstm1)
    
    input_comment = Input(shape=(comm_len,))
    embedding2 = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=comm_len, trainable=False)(input_comment)
    bilstm2 = Bidirectional(LSTM(10, return_sequences=True))(embedding2)
#    dense2 = Dense(10, activation='sigmoid')(bilstm2)
        
    x = concatenate([bilstm1, bilstm2], axis=1)
    x = Dense(10, activation='sigmoid')(x)
    x = Bidirectional(LSTM(10))(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[input_parent, input_comment], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print(model.summary())
    return model

#input_politics["comment"] = input_politics["comment"].apply(lambda x: Utils.cleanup_str(x))
#input_politics["parent_comment"] = input_politics["parent_comment"].apply(lambda x: Utils.cleanup_str(x))

#tok, vocab_size = Utils.init_tokenizer(input_politics["comment"])

#a, b = Utils.encode_docs(tok, input_politics)
#xx = np.concatenate((a,b), axis=1)
#np.array_equal(a, xx[:,:559])
#np.array_equal(b, xx[:,559:1118])
#x_train, x_test, y_train, y_test = train_test_split(xx, y, test_size=0.2, random_state=0)

#c, d = Utils.encode_docs1(tok, input_politics, "comment")
    
vocab_size = 0
max_len = 0 

def initz():
    input_politics = load_filtered_dataset("/home/shariq/MSc/Research/dataset/train-politics.csv")
    input_politics["all_comments"] = input_politics["parent_comment"].map(str) + " " + input_politics["comment"]
    return input_politics

def main():
    global vocab_size, max_len, tok, embedding_matrix
    
    input_politics = initz()
    pick_col = "all_comments"
    tok, vocab_size = Utils.init_tokenizer(input_politics[pick_col])
    max_len, d = Utils.encode_docs1(tok, input_politics, pick_col)
    y = np.array(input_politics["label"])
    
    print('Max length of sequences : %s' % max_len)
    
    embedding_matrix = Utils.create_embeddings(vocab_size, tok)
    x_train, x_test, y_train, y_test = train_test_split(d, y, test_size=0.2, random_state=0)

    #model = base_model(vocab_size, embedding_matrix, max_len)
    model = base_model()
    model.fit(x_train, y_train, epochs=5, batch_size=32)
    loss, accuracy = model.evaluate(x_test, y_test)
    print('Accuracy: %f' % (accuracy*100))
    
def multi_input():
    global vocab_size, embedding_matrix
    input_politics = initz()
    pick_col = "all_comments"
    tok, vocab_size = Utils.init_tokenizer(input_politics[pick_col])
    
    max_len_comm, en_comm = Utils.encode_docs1(tok, input_politics, "comment")
    max_len_pr_comm, en_pr_comm = Utils.encode_docs1(tok, input_politics, "parent_comment")
    
    embedding_matrix = Utils.create_embeddings(vocab_size, tok)
    
    y = np.array(input_politics["label"])
    x = np.concatenate((en_pr_comm, en_comm), axis=1)
    
    np.array_equal(en_pr_comm, x[:,:max_len_pr_comm])
    np.array_equal(en_comm, x[:,max_len_pr_comm:max_len_pr_comm+max_len_comm])
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    multi_model = multi_input_model(max_len_pr_comm, max_len_comm)
    multi_model.fit([x_train[:,:max_len_pr_comm], x_train[:,max_len_pr_comm:max_len_pr_comm+max_len_comm]],
                    y_train, epochs=1, batch_size=128)
    

if __name__ == "__main__":
    #main()
    multi_input()
  