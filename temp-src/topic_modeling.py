#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: shariq
"""

import spotlight

filterx = {'types': "DBpedia:SportsTeam"}


str1 = "This is going to be the most unorthodox presidency of all time. Certainly the most corrupt."
str2 = "Sky In Italy report that negotiations between Napoli and Chelsea over Maurizio Sarri's appointment as Conte's potential replacement are in deadlock."
annotations = spotlight.annotate('http://localhost/rest/annotate',
                                  str2,
                                  confidence=0.4, support=20, filters=filterx)

print(annotations)

type(annotations)
type(annotations[0]['types'])

print(str(annotations))

type_arr = [y['types'] for y in annotations]
len(type_arr)
print(type_arr)
type(type_arr)

all = set()
for f in type_arr:
    arr = [f1 for f1 in f.split(',')]
    for f2 in arr:
        if(f2.startswith('DBpedia')):
            all.add(f2)

print(all)      
                

print(str(annotations[0]['types']))

print((annotations[0]["types"]))

for my_var in annotations:
    print str(my_var['types'])
    
##### Read CSV data & annotate    

import csv
import sys

data = []
num = 0

with open("/home/shariq/MSc/Research/dataset/reddit/pol/train-balanced.csv") as f:
    row = csv.reader(f, delimiter='\t')
    for r in row:
        data.append(r)
        num += 1
        if num > 100:
            break;
            
            
print(data)            
type(data)
print(data[0])
print(data[:][3])

subredit = [x[3] for x in data]
print(subredit)

politics_filter = {'types' : "DBpedia:Politician," +
                   "DBpedia:PoliticalParty," +
                   "DBpedia:OfficeHolder," +
                   "DBpedia:PoliticalParty," +
                   "DBpedia:GovernmentAgency"}

topics = []
n = 0
valid_cnt = 0 
for d in data:
    all_type = set()
    n += 1
    para = d[9] + " " + d[1]
    try: 
        temp = spotlight.annotate('http://localhost/rest/annotate', para, 
                                  confidence=0.3, filters=politics_filter)
    except:
        sys.exc_clear()
    else:     
        valid_cnt += 1 
        type_arr = [y['types'] for y in temp]
        for f in type_arr:
            arr = [f1 for f1 in f.split(',')]
            for f2 in arr:
                if(f2.startswith('DBpedia')):
                    all_type.add(f2)
        print(str(valid_cnt) + " - " + str(n) + " : " + str(para) + " => " + str(all_type) + "\n")           
        
        
##### Pandas

import pandas as pd

input = pd.read_csv("/home/shariq/MSc/Research/dataset/reddit/pol/train-balanced.csv", sep='\t',
                    names = ["label", "comment", "auth", "subreddit", "score", "ups", "downs", "date", "created_utc", "parent_comment"])      

type(input)

input.columns.tolist()

input.at[1,"comment"]

input[["label", "comment", "parent_comment"]]

input.loc[:, ["label", "comment", "parent_comment"]]

for index, row in input.iterrows():
    print index, row["parent_comment"], row["comment"]
    
    
for index, row in input.iterrows():
    print(row["parent_comment"] + row["comment"])


for index, row in input.iterrows():
    spotlight.annotate('http://localhost/rest/annotate', row["parent_comment"] + " " + row["comment"], 
                                  confidence=0.3, filters=politics_filter)




col_names={"label", "comment", "parent_comment"}
politics_data = pd.DataFrame(columns=col_names)
politics_data.columns.tolist()

n=0
for index, row in input.iterrows():
    #all_type = set()
    para = row["parent_comment"] + " " + row["comment"]
    #n += 1
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



####

topics = []
n = 0
valid_cnt = 0 
all_type = set()

for index, row in input.iterrows():
    #all_type = set()
    para = row["parent_comment"] + " " + row["comment"]
    n += 1
    try: 
        temp = spotlight.annotate('http://localhost/rest/annotate', para, 
                                  confidence=0.0, filters=politics_filter)
    except:
        sys.exc_clear()
    else:     
        valid_cnt += 1 
        type_arr = [y['types'] for y in temp]
        for f in type_arr:
            arr = [f1 for f1 in f.split(',')]
            for f2 in arr:
                if(f2.startswith('DBpedia')):
                    all_type.add(f2)
        #print(str(valid_cnt) + " - " + str(n) + " : " + str(para) + " => " + str(all_type) + "\n")     
#print(str(valid_cnt) + "\n")     
    
print(all_type)

##### Genism

politics_data["content"] = politics_data["parent_comment"].astype(str) + " " + politics_data["comment"]   

import gensim 

def preprocess(raw_df):
    for i, row in raw_df.iterrows():
        temp_content = row['comment']
        yield gensim.utils.simple_preprocess (temp_content)

documents = []
documents = list(preprocess(politics_data))

type(documents)
print(len(documents))
print(documents[1])
print(len(documents[1]))


def find_max_list(list): 
    list_len = [len(i) for i in list] 
    print(max(list_len))
    
find_max_list(documents)    


model = gensim.models.Word2Vec (documents, size=150, window=10, min_count=2, workers=10)
model.train(documents,total_examples=len(documents),epochs=10)    
    

import os
os.getcwd()
os.chdir('/home/shariq')
import pickle

f = open('store.pckl', 'wb')
pickle.dump(input, f)
pickle.dump(politics_data, f)
pickle.dump(documents, f)
f.close()

f = open('store.pckl', 'rb')
obj = pickle.load(f)
f.close()
