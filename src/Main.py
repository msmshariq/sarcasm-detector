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

from TopicExtractor import TopicExtractor
from ModelBuilder import ModelBuilder
import Utils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def load_raw_dataset(file_path):
    all_columns = ["label", "comment", "auth", "subreddit", "score", "ups", 
                   "downs", "date", "created_utc", "parent_comment"]
    return pd.read_csv(file_path, sep='\t',names = all_columns)     
        

def load_filtered_dataset(file_path):
    return pd.read_csv(file_path, sep='\t', names = ["label", "comment", "parent_comment"]) 
    

def get_politics_data(input):
    politics_filter = {'types' : "DBpedia:Politician," +
                   "DBpedia:PoliticalParty," +
                   "DBpedia:OfficeHolder," +
                   "DBpedia:PoliticalParty," +
                   "DBpedia:GovernmentAgency"}

    col_names=["label", "comment", "parent_comment"]
    politcs_extract = TopicExtractor(confidence=0.0, filter=politics_filter, columns=col_names)
#    return politcs_extract
    return politcs_extract.filter_dataset(input)

def init_test_dataset(file_path):
    all_test_data = load_raw_dataset(file_path)
    test_politics = get_politics_data(all_test_data)
    return test_politics
    
def save_to_file(file_path, dataframe):
    dataframe.to_csv(file_path, sep='\t', index=False, header=False)
    
def test_multi_input_model(modle_path, pr_comm_len, comm_len):
    politics_test = init_test()
    # TODO: fix length
    en_t_comm = Utils.encode_test_docs(tok, politics_test, 'comment', comm_len)
    en_t_pr_comm = Utils.encode_test_docs(tok, politics_test, 'parent_comment', pr_comm_len)
    y_ = np.array(politics_test["label"])
    y = to_categorical(y_ ,num_classes = None)
    x = np.concatenate((en_t_pr_comm, en_t_comm), axis=1)
    # load model from file
    # TODO: read file name from file
    multi_input=load_model(modle_path)
    loss, accuracy = multi_input.evaluate([x[:, :len_pr_comm], x[:, len_pr_comm:len_pr_comm + len_comm]], y)
    print('Accuracy: %f' % (accuracy*100))
    print('Loss: %f' % (loss*100))
    preds = multi_input.predict([x[:, :len_pr_comm], x[:, len_pr_comm:len_pr_comm + len_comm]])
    pred_classes = np.argmax(preds, axis=1)
    return pred_classes, y_

#input_politics["comment"] = input_politics["comment"].apply(lambda x: Utils.cleanup_str(x))
#input_politics["parent_comment"] = input_politics["parent_comment"].apply(lambda x: Utils.cleanup_str(x))

vocab_size = 0
max_len = 0 

def initz():
    input_politics = load_filtered_dataset("/home/shariq/MSc/Research/dataset/train-politics.csv")
    input_politics["parent_comment"] = input_politics["parent_comment"].apply(lambda x: Utils.cleanup_str(x))
    input_politics["parent_comment"] = input_politics["parent_comment"].apply(lambda x: Utils.lemmatize_str(x))
    input_politics["comment"] = input_politics["comment"].apply(lambda x: Utils.lemmatize_str(x))
    input_politics["all_comments"] = input_politics["parent_comment"].map(str) + " " + input_politics["comment"]
    return input_politics

def init_test():
    input_politics_test = load_filtered_dataset("/home/shariq/MSc/Research/dataset/test-politics.csv")
    input_politics_test["parent_comment"] = input_politics_test["parent_comment"].apply(lambda x: Utils.cleanup_str(x))
    input_politics_test["parent_comment"] = input_politics_test["parent_comment"].apply(lambda x: Utils.lemmatize_str(x))
    input_politics_test["comment"] = input_politics_test["comment"].apply(lambda x: Utils.lemmatize_str(x))
    input_politics_test["all_comments"] = input_politics_test["parent_comment"].map(str) + " " + input_politics_test["comment"]
    return input_politics_test


if __name__ == "__main__":
#    main()
#    multi_input()
#    pol_test = init_test_dataset('/home/shariq/MSc/Research/dataset/reddit/pol/test-balanced.csv')

    global vocab_size, embedding_matrix, multi_model, len_comm, len_pr_comm
    input_politics = initz()
    pick_col = "all_comments"
    tok, vocab_size = Utils.init_tokenizer(input_politics[pick_col])
    
    len_comm, en_comm = Utils.encode_docs1(tok, input_politics, "comment")
    len_pr_comm, en_pr_comm = Utils.encode_docs1(tok, input_politics, "parent_comment")
    
    embedding_matrix = Utils.create_embeddings(vocab_size, tok)
    
    y = np.array(input_politics["label"])
    x = np.concatenate((en_pr_comm, en_comm), axis=1)
    y = to_categorical(y ,num_classes = None)

    
    np.array_equal(en_pr_comm, x[:, :len_pr_comm])
    np.array_equal(en_comm, x[:, len_pr_comm:len_pr_comm+ len_comm])
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
#    tb_callback = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=128, write_graph=True, 
#                                  write_grads=True, write_images=True, embeddings_freq=0, 
#                                  embeddings_layer_names=None, embeddings_metadata=None, 
#                                  embeddings_data=None)
#    
#    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
#                              patience=2, min_lr=0.0, verbose=1)
    
#    callbacks = [tb_callback]
#    builder = ModelBuilder(vocab_size, embedding_matrix=embedding_matrix)   
#    model = builder.multi_input_model_x(pr_len=len_pr_comm, comm_len=len_comm, optimizer='adam')
#    model.fit([x_train[:, :len_pr_comm], x_train[:, len_pr_comm:len_pr_comm + len_comm]],
#              y_train, epochs=25, batch_size=128, callbacks=callbacks, 
#              validation_data=([x_test[:, :len_pr_comm], x_test[:, len_pr_comm:len_pr_comm + len_comm]], 
#                               y_test))
        
    
    for f in ['rmsprop']:
        builder = ModelBuilder(vocab_size, embedding_matrix=embedding_matrix)
        model = builder.multi_input_model(pr_len=len_pr_comm, comm_len=len_comm, optimizer=f)
#        filepath = 'model-latest-' + f + '-opt.h5'
        logpath = 'logs-latest-1' + f
        
#        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
        reduce_lr_acc = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                              patience=2, min_lr=0.0, verbose=1)
        reduce_lr_val = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.0, verbose=1)
        
#        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-6)
#        early_stopper = EarlyStopping(min_delta=0.001, patience=2)
        tb_callback = TensorBoard(log_dir=logpath, histogram_freq=1, batch_size=128, write_graph=True, 
                                  write_grads=True, write_images=True, embeddings_freq=0, 
                                  embeddings_layer_names=None, embeddings_metadata=None, 
                                  embeddings_data=None)
        callbacks_list = [reduce_lr_acc, reduce_lr_val, tb_callback]
        model.fit([x_train[:, :len_pr_comm], x_train[:, len_pr_comm:len_pr_comm + len_comm]],
                    y_train, epochs=20, batch_size=128, callbacks=callbacks_list,
                    validation_data=([x_test[:, :len_pr_comm], x_test[:, len_pr_comm:len_pr_comm + len_comm]], 
                               y_test))
        
    
#    pred_classes, classes = test_multi_input_model('../models/model-rmsprop-opt.h5', len_pr_comm, len_comm) 
#    confusion_matrix(classes, pred_classes)       
#    f1_score(classes, pred_classes)      
#
#    
#    politics_test = init_test()
#    # TODO: fix length
#    en_t_comm = Utils.encode_test_docs(tok, politics_test, 'comment', len_comm)
#    en_t_pr_comm = Utils.encode_test_docs(tok, politics_test, 'parent_comment', len_pr_comm)
#    y_ = np.array(politics_test["label"])
#    y1 = to_categorical(y_ ,num_classes = None)
#    x1 = np.concatenate((en_t_pr_comm, en_t_comm), axis=1)
#    
#    model.evaluate([x1[:, :len_pr_comm], x1[:, len_pr_comm:len_pr_comm + len_comm]], y1)
#    
#    preds = model.predict([x1[:, :len_pr_comm], x1[:, len_pr_comm:len_pr_comm + len_comm]])
#    pred_classes = np.argmax(preds, axis=1)
#    confusion_matrix(y_, pred_classes)       
#    f1_score(y_, pred_classes)   

