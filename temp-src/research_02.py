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
from keras.layers import Embedding, Dense, Flatten, Dropout, Input, LSTM, TimeDistributed, SpatialDropout1D, Bidirectional, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split


input_politics = pd.read_csv("/home/shariq/MSc/Research/dataset/train-politics.csv", sep='\t',
                    names = ["label", "comment", "parent_comment"])      

# labels
y = np.array(input_politics["label"])

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(input_politics["comment"])
vocab_size = len(t.word_index) + 1
#print(vocab_size)

encoded_docs = t.texts_to_sequences(input_politics["comment"])
#print(encoded_docs)
type(encoded_docs)
#print len(max(encoded_docs, key=len))

max_length = len(max(encoded_docs, key=len))
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
#print(padded_docs)

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
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
        
#print(embedding_matrix[1])
#len(embedding_matrix)

# test train split
x_train, x_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.2, random_state=0)


# define model
#model.add(Flatten())
#model.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2))
model.add(TimeDistributed(Dense(32, activation='sigmoid')))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

#model.add(Dense(32, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(x_train, y_train, epochs=10, batch_size=128)
# evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %f' % (accuracy*100))

###
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
###


length=257
inputs1 = Input(shape=(length,))
#embedding1 = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=257)(inputs1)
embedding1 = Embedding(vocab_size, 100, input_length=257, trainable=False)(inputs1)
conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
drop1 = Dropout(0.5)(conv1)
pool1 = MaxPooling1D(pool_size=2)(drop1)
flat1 = Flatten()(pool1)
# channel 2
inputs2 = Input(shape=(length,))
#embedding2 = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=257)(inputs2)
embedding2 = Embedding(vocab_size, 100, input_length=257, trainable=False)(inputs2)
conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
drop2 = Dropout(0.5)(conv2)
pool2 = MaxPooling1D(pool_size=2)(drop2)
flat2 = Flatten()(pool2)
# channel 3
inputs3 = Input(shape=(length,))
#embedding3 = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=257)(inputs3)
embedding3 = Embedding(vocab_size, 100, input_length=257, trainable=False)(inputs3)
conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
drop3 = Dropout(0.5)(conv3)
pool3 = MaxPooling1D(pool_size=2)(drop3)
flat3 = Flatten()(pool3)
# merge
merged = concatenate([flat1, flat2, flat3])
# interpretation
dense1 = Dense(10, activation='relu')(merged)
outputs = Dense(1, activation='sigmoid')(dense1)
model_conv = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
# compile
model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize
print(model_conv.summary())

SVG(model_to_dot(model_conv).create(prog='dot', format='svg'))

model_conv.fit([x_train,x_train,x_train], y_train, epochs=10)

loss_conv, accuracy_conv = model_conv.evaluate([x_test,x_test,x_test], y_test, verbose=0)
print('Accuracy: %f' % (accuracy_conv*100))

#########

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, random_state=42))
])
                     
x_train1, x_test1, y_train1, y_test1 = train_test_split(input_politics["comment"], y, test_size=0.2, random_state=0)

    
text_clf.fit(x_train1, y_train1) 
predicted = text_clf.predict(x_test1)
np.mean(predicted == y_test1)            

from sklearn.model_selection import GridSearchCV

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(x_train1, y_train1)
print(gs_clf.best_score_)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    

#########
    
model_x = Sequential()
model_x.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=257, trainable=False))
#model_x.add(Flatten())
#model_x.add(SpatialDropout1D(0.4))
#model_x.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2))
#model_x.add(LSTM(32, return_sequences=True))
model_x.add(LSTM(16, return_sequences=True))
model_x.add(LSTM(16))
model_x.add(Dense(1,activation='softmax'))
model_x.compile(loss = 'binary_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])
print(model_x.summary())
    
SVG(model_to_dot(model_x).create(prog='dot', format='svg'))

model_x.fit(x_train, y_train, epochs=5)

loss_x, accuracy_x = model_conv.evaluate([x_test,x_test,x_test], y_test, verbose=0)
print('Accuracy: %f' % (accuracy_conv*100))
    
    
    
##########


    
from keras.wrappers.scikit_learn import KerasClassifier

def create_model():    
    model1 = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=257, trainable=False)
    model1.add(e)
    model1.add(Flatten())
    model1.add(Dense(12, activation='sigmoid'))
    model1.add(Dense(1, activation='sigmoid'))
    # compile the model
    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model1
    
    # summarize the model

model1 = KerasClassifier(build_fn=create_model, verbose=0)

    
print(model1.summary())
# fit the model
model1.fit(x_train, y_train, epochs=10)
# evaluate the model
loss1, accuracy1 = model1.evaluate(x_test, y_test)
print('Accuracy: %f' % (accuracy1*100))    

batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model1, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x_test[:250], y_test[:250])
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))




#########

model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=257, trainable=False)
model.add(e)

model.add(Bidirectional(LSTM(10, return_sequences=True)))
model.add(Dense(10, activation='sigmoid'))

#model.add(Bidirectional(LSTM(10, return_sequences=True)))
#model.add(Dense(10, activation='sigmoid'))

model.add(Bidirectional(LSTM(10)))
model.add(Dense(1))
#model.add(Activation('softmax')) ## NOT THAT GREAT
model.add(Activation('relu')) ## BETTER
#model.add(Activation('sigmoid')) ## NOT THAT GREAT
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

model.fit(x_train, y_train, epochs=5, batch_size=16, shuffle=False)
# evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %f' % (accuracy*100))
