#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:42:52 2018

@author: shariq
"""
import numpy as np
np.random.seed(1987)

from tensorflow import set_random_seed
set_random_seed(1987)

import random as rn
rn.seed(1987) 

from TopicExtractor import TopicExtractor
from ModelBuilder import ModelBuilder
import Utils
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.utils import to_categorical


def load_raw_dataset(file_path):
    all_columns = ["label", "comment", "auth", "subreddit", "score", "ups", 
                   "downs", "date", "created_utc", "parent_comment"]
    return pd.read_csv(file_path, sep='\t',names = all_columns)     
        

def load_filtered_dataset(file_path):
    return pd.read_csv(file_path, sep='\t', names = ["label", "comment", "parent_comment"]) 
    

def get_politics_data(input):
    politics_filter = {"types" : "DBpedia:Politician," +
                       "DBpedia:PoliticalParty," +
                       "DBpedia:OfficeHolder," +
                       "DBpedia:PoliticalParty," +
                       "DBpedia:GovernmentAgency"}

    col_names=["label", "comment", "parent_comment"]
    politcs_extract = TopicExtractor(confidence=0.0, filter=politics_filter, columns=col_names)
    return politcs_extract.filter_dataset(input)

def get_sports_data(input):
    politics_filter = {"types" : "DBpedia:Sport," + 
                       "DBpedia:SportsClub," +
                       "DBpedia:SportsLeague," + 
                       "DBpedia:SportsTeam," + 
                       "DBpedia:MotorsportRacer," + 
                       "DBpedia:WinterSportPlayer," +
                       "DBpedia:SportsTeamMember," + 
                       "DBpedia:SportsManager," +
                       "DBpedia:SportsEvent," + 
                       "DBpedia:SportFacility," +
                       "DBpedia:SportCompetitionResult," + 
                       "DBpedia:SportsSeason"}

    col_names=["label", "comment", "parent_comment"]
    sports_extract = TopicExtractor(confidence=0.7, filter=politics_filter, columns=col_names)
#    return politcs_extract
    return sports_extract.filter_dataset(input)

def init_test_dataset(file_path):
    all_test_data = load_raw_dataset(file_path)
    test_politics = get_politics_data(all_test_data)
    return test_politics
    
def save_to_file(file_path, dataframe):
    dataframe.to_csv(file_path, sep='\t', index=False, header=False)
    
def init_filtered_data(file_path):
    filtered_data = load_filtered_dataset(file_path)
    filtered_data["parent_comment"] = filtered_data["parent_comment"].apply(lambda x: Utils.cleanup_str(x))
    filtered_data["parent_comment"] = filtered_data["parent_comment"].apply(lambda x: Utils.lemmatize_str(x))
    filtered_data["comment"] = filtered_data["comment"].apply(lambda x: Utils.lemmatize_str(x))
    filtered_data["all_comments"] = filtered_data["parent_comment"].map(str) + " " + input_politics["comment"]
    return filtered_data    

def train_model(model, x_data, y_data, epochs, tag):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
    
    # define callbacks used during the training
    tb_callback = TensorBoard(log_dir="./logs/model-{}".format(tag), histogram_freq=1, 
                              batch_size=128, write_graph=True, write_grads=True, write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=2, min_lr=0.0, verbose=1)
    checkpoint = ModelCheckpoint("models/model-{}.h5".format(tag), 
                                 monitor="val_acc", verbose=1, save_best_only=False, mode="max")
    callbacks_list = [tb_callback, reduce_lr, checkpoint]
    
    # train the model
    model.fit([x_train[:, :len_pr_comm], x_train[:, len_pr_comm:len_pr_comm + len_comm]], y_train, 
              epochs=epochs, batch_size=128, callbacks=callbacks_list, 
              validation_data=([x_test[:, :len_pr_comm],x_test[:, len_pr_comm:len_pr_comm + len_comm]], y_test))
    
    
def test_model(modle, x, y):
    loss, accuracy = model.evaluate([x[:, :len_pr_comm], x[:, len_pr_comm:len_pr_comm + len_comm]], y)
    print('Accuracy: %f' % (accuracy*100))
    print('Loss: %f' % (loss*100))
    preds = model.predict([x[:, :len_pr_comm], x[:, len_pr_comm:len_pr_comm + len_comm]])
    pred_classes = np.argmax(preds, axis=1)
    print(confusion_matrix(y, pred_classes))    
    f1_score((y, pred_classes))


vocab_size = 0
max_len = 0 


def init_test():
    input_politics_test = load_filtered_dataset("/home/shariq/MSc/Research/dataset/test-politics.csv")
#    input_politics_test = load_filtered_dataset("/home/shariq/MSc/Research/dataset/reddit/main/train-sports.balanced.csv")
#    input_politics_test = input_politics_test[7001:10001]
    input_politics_test["parent_comment"] = input_politics_test["parent_comment"].apply(lambda x: Utils.cleanup_str(x))
    input_politics_test["parent_comment"] = input_politics_test["parent_comment"].apply(lambda x: Utils.lemmatize_str(x))
    input_politics_test["comment"] = input_politics_test["comment"].apply(lambda x: Utils.lemmatize_str(x))
    input_politics_test["all_comments"] = input_politics_test["parent_comment"].map(str) + " " + input_politics_test["comment"]
    return input_politics_test


if __name__ == "__main__":
    choice = sys.argv[0]
    #init training on politics data
    input_politics = init_filtered_data("/home/shariq/MSc/Research/dataset/train-politics.csv")
    tok, vocab_size = Utils.init_tokenizer(input_politics["all_comments"])
    
    global len_comm, len_pr_comm
    len_comm, en_comm_len = Utils.encode_docs(tok, input_politics, "comment")
    len_pr_comm, en_pr_comm_len = Utils.encode_docs(tok, input_politics, "parent_comment")
    
    embedding_matrix = Utils.create_embeddings(vocab_size, tok)
    
    if choice == "train":
        y = np.array(input_politics["label"])
        y = to_categorical(y ,num_classes = None)
        x = np.concatenate((en_pr_comm_len, en_comm_len), axis=1)
        
        for opt in ["adam", "rmsprop"]:
            for loss in ["binary_crossentropy", "mse"]:
                path = opt + "-" + loss
                builder = ModelBuilder(vocab_size = vocab_size, embedding_matrix = embedding_matrix)
                model = builder.multi_input_model(len_pr_comm, len_comm, optimizer=opt, loss=loss)
                train_model(model, x, y, 20)

    
#    np.array_equal(en_pr_comm, x[:, :len_pr_comm])
#    np.array_equal(en_comm, x[:, len_pr_comm:len_pr_comm+ len_comm])
   
    if choice == "test":
        politics_test = init_filtered_data("/home/shariq/MSc/Research/dataset/test-politics.csv")
        y_ = np.array(politics_test["label"])
        y = to_categorical(y_ ,num_classes = None)
        
        en_test_comm = Utils.encode_test_docs(tok, politics_test, 'comment', en_comm_len)
        en_test_pr_comm = Utils.encode_test_docs(tok, politics_test, 'parent_comment', en_pr_comm_len)
        x = np.concatenate((en_test_pr_comm, en_test_comm), axis=1)



        
#   ******************************************************S 
    
#    pred_classes, classes = test_multi_input_model('../models/model-rmsprop-opt.h5', len_pr_comm, len_comm) 
#    confusion_matrix(classes, pred_classes)       
#    f1_score(classes, pred_classes)      
#
    
#    sports_test = init_test()
#    en_t_comm = Utils.encode_test_docs(tok, sports_test, 'comment', len_comm)
#    en_t_pr_comm = Utils.encode_test_docs(tok, sports_test, 'parent_comment', len_pr_comm)
#    y_ = np.array(sports_test["label"])
#    y2 = to_categorical(y_ ,num_classes = None)
#    x2 = np.concatenate((en_t_pr_comm, en_t_comm), axis=1)
#
#    model.evaluate([x2[:, :len_pr_comm], x2[:, len_pr_comm:len_pr_comm + len_comm]], y2)
#    
#    preds = model.predict([x2[:, :len_pr_comm], x2[:, len_pr_comm:len_pr_comm + len_comm]])
#    pred_classes = np.argmax(preds, axis=1)
#    confusion_matrix(y_, pred_classes)       
#    f1_score(y_, pred_classes)
    
    
#    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.1, random_state=0)

    
#    new_model.fit([x_train[:, :len_pr_comm], x_train[:, len_pr_comm:len_pr_comm + len_comm]],
#                   y_train, epochs=20, batch_size=128)
    
#    mod.evaluate([x_test[:, :len_pr_comm], x_test[:, len_pr_comm:len_pr_comm + len_comm]], y_test)
#              
    
    
#    new_model.evaluate([x1[:, :len_pr_comm], x1[:, len_pr_comm:len_pr_comm + len_comm]], y1)
#    mod.evaluate([x1[:, :len_pr_comm], x1[:, len_pr_comm:len_pr_comm + len_comm]], y1)
#    
#    preds = model.predict([x1[:, :len_pr_comm], x1[:, len_pr_comm:len_pr_comm + len_comm]])
#    pred_classes = np.argmax(preds, axis=1)
#    confusion_matrix(y_, pred_classes)       
#    f1_score(y_, pred_classes)
            
#    y_classes = keras.np_utils.probas_to_classes(y_proba)