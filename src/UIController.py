#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: shariq
"""

import Main
import Utils
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request

app = Flask(__name__)
 
# Use the politics data to initilize the tokenizer
input_politics = Main.init_filtered_data("/home/shariq/MSc/Research/dataset/train-politics.csv")
tok, vocab_size = Utils.init_tokenizer(input_politics["all_comments"])
len_comm = 257
len_pr_comm = 559

model1 = load_model('../models/model-adam-opt.h5')
model2 = load_model('../models/model-rmsprop-opt.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_politics():
    parentComment = request.form['parentComment']
    comment = request.form['comment']
    model_select = request.form['sel-model']
    
    encoded_pr_comm = tok.texts_to_sequences([parentComment]) 
    paded_pr_comm = pad_sequences(encoded_pr_comm, maxlen=len_pr_comm,
                                    padding='post', truncating='post')
    
    encoded_comm = tok.texts_to_sequences([comment]) 
    paded_comm = pad_sequences(encoded_comm, maxlen=len_comm,
                                    padding='post', truncating='post')
     
    x = np.concatenate((paded_pr_comm, paded_comm), axis=1)
    
    if model_select == 'Model-1':
        preds = model1.predict([x[:, :len_pr_comm], x[:, len_pr_comm:len_pr_comm + len_comm]])
    elif model_select == 'Model-2':
        preds = model2.predict([x[:, :len_pr_comm], x[:, len_pr_comm:len_pr_comm + len_comm]])
        
    print(preds)
    pred_class = np.argmax(preds, axis=1)
    
    if pred_class[0] == 1:
        return "Sarcastic!"
    elif pred_class[0] == 0:
        return "Not Sarcastic"
    
if __name__ == "__main__":
    app.run(host='0.0.0.0')
