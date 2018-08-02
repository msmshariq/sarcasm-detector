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
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.models import Model
from keras.layers import Embedding, Dense, Input, LSTM, Bidirectional, Activation
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

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

def main():
    global vocab_size, max_len, tok, embedding_matrix
    input_politics = load_filtered_dataset("/home/shariq/MSc/Research/dataset/train-politics.csv")
    input_politics["all_comments"] = input_politics["parent_comment"].map(str) + " " + input_politics["comment"]
    pick_col = "comment"
    tok, vocab_size = Utils.init_tokenizer(input_politics[pick_col])
    max_len, d = Utils.encode_docs1(tok, input_politics, pick_col)
    y = np.array(input_politics["label"])
    
    print('Max length of sequences : %s' % max_len)
    
    embedding_matrix = Utils.create_embeddings(vocab_size, tok)
    x_train, x_test, y_train, y_test = train_test_split(d, y, test_size=0.2, random_state=0)

    
#    model = base_model(vocab_size, embedding_matrix, max_len)
#    model.fit(x_train, y_train, epochs=5, batch_size=32)
#    loss, accuracy = model.evaluate(x_test, y_test)
#    print('Accuracy: %f' % (accuracy*100))
    
    model = KerasClassifier(build_fn=base_model, verbose=1)
    # define the grid search parameters
    batch_size = [32, 64, 128, 256]
    epochs = [5, 10, 15]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(x_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

if __name__ == "__main__":
    main()