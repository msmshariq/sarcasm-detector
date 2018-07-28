#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: shariq
"""

import spotlight
import pandas as pd
import sys
import gensim 
import numpy as np

politics_filter = {'types' : "DBpedia:Politician," +
                   "DBpedia:PoliticalParty," +
                   "DBpedia:OfficeHolder," +
                   "DBpedia:PoliticalParty," +
                   "DBpedia:GovernmentAgency"}


input = pd.read_csv("/home/shariq/MSc/Research/dataset/reddit/pol/train-balanced.csv", sep='\t',
                    names = ["label", "comment", "auth", "subreddit", "score", "ups", "downs", 
                             "date", "created_utc", "parent_comment"])      


col_names={"label", "comment", "parent_comment"}
politics_data = pd.DataFrame(columns=col_names)
politics_data.columns.tolist()

n=0
for index, row in input.iterrows():
    para = row["parent_comment"] + " " + row["comment"]
    try: 
        temp = spotlight.annotate('http://localhost/rest/annotate', para, 
                                  confidence=0.0, filters=politics_filter)
    except:
        sys.exc_clear()
    else:     
        n += 1
        if (n % 100 == 0):
            print ("read {0} reviews".format (n))
        politics_data = politics_data.append({'label': row['label'], 
                              'comment': row['comment'], 
                              'parent_comment': row['parent_comment']}, ignore_index=True)


politics_data = politics_data[["label", "comment", "parent_comment"]]
politics_data.columns.tolist()
type(politics_data)

max_words = politics_data["comment"].str.split().map(len).max()

#politics_data["comment"].apply(lambda x: gensim.utils.simple_preprocess(x))

#politics_data.to_csv('/home/shariq/MSc/Research/dataset/train-politics.csv', sep='\t')

#politics_data = pd.read_csv("/home/shariq/MSc/Research/dataset/train-politics.csv", sep='\t',
#                    names = ["label", "comment", "parent_comment"])  

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten

politics_data["comment_process"] = politics_data["comment"].apply(lambda x: " ".join(gensim.utils.simple_preprocess(x)))


politics_data["comment_rm_stp"] = politics_data["comment_process"].apply(lambda x: gensim.parsing.remove_stopwords(x))

#politics_data["comment_clean"] = politics_data["comment_rm_stp"].apply(lambda x: " ".join(gensim.utils.simple_preprocess(x)))
politics_data["comment_clean"] = politics_data["comment_rm_stp"]


print(politics_data["comment_rm_stp"])
print(politics_data["comment_clean"])


tok = Tokenizer()
tok.fit_on_texts(politics_data["comment_clean"])
vocab_size = len(tok.word_index) + 1
print(vocab_size)


encoded_docs = tok.texts_to_sequences(politics_data["comment_clean"])
tok.word_index
print(encoded_docs)
type(encoded_docs)
print len(max(encoded_docs, key=len))

max_length = len(max(encoded_docs, key=len))
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

embeddings_index = dict()
f = open('/home/shariq/MSc/Research/glove.6B/glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tok.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

y = np.array(politics_data["label"])

# define model
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=103, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, y, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, y, verbose=0)
print('Accuracy: %f' % (accuracy*100))

# ****

def preprocess(raw_df):
    for i, row in raw_df.iterrows():
        temp_content = row['comment']
        yield gensim.utils.simple_preprocess (temp_content)

documents = []
documents = list(preprocess(politics_data))
    
print(documents)


type(documents[0])
lenx=0
for line in documents:
    tmp_len=len(line)
    if tmp_len>lenx:
        lenx=tmp_len

print(lenx)

model = gensim.models.Word2Vec(documents, size=200, window=10, min_count=1, workers=10)
#model = gensim.models.Word2Vec(documents)

model.train(documents,total_examples=len(documents),epochs=10)   




doc_df = pd.DataFrame(preprocess(politics_data))
print(doc_df)

doc_df = doc_df.fillna(0.0)
doc_df        
type(doc_df)



doc_nparr = doc_df.values
type(doc_nparr)        
doc_nparr

def vectorize_str(str_arr):
    tmp_arr = []
    for word in str_arr:
        if word == 0.0:
            tmp_arr.append(0.0)
        else: 
            tmp_arr.append(np.average(model.wv.word_vec(word)))
    return tmp_arr

   



print(vectorize_str(doc_nparr[1]))
print(len(vectorize_str(doc_nparr[1])))

x = np.concatenate([[vectorize_str(line) for line in doc_nparr]], axis=0)
len(x)
print(x)
print(x[2])
print(x[1])
print(doc_nparr[1])
type(x)

y = np.concatenate([politics_data["label"].apply(lambda x: [0,1] if x == 1 else [1,0])])
len(y)
y[7]
politics_data["label"]

#x = np.array([[np.average(model.wv.word_vec(word)) for word in sentence] for sentence in doc_nparr])


w1="nice"
model.wv.most_similar(positive=w1)

word_vector = model.wv
type(model)
type(model.wv.word_vec("dirty"))
np.average(model.wv.word_vec("clean"))

model.wv.most_similar(positive='clean')

### Keras implementation ###

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

len(x_train)
len(y_train)
len(x_test)
len(y_test)

keras_model = Sequential()
keras_model.add(Dense(units=64, activation='relu', input_dim=240))
keras_model.add(Dense(units=1, activation='softmax'))
 
keras_model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


keras_model.fit(x_train, y_train, epochs=5)
keras_model.fit(list(x_train), list(y_train), epochs=5)

keras_model.fit()




