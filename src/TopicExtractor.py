#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: shariq
"""

import spotlight
import sys
import pandas as pd
import logging as log

class TopicExtractor:

    spotlight_endpoint = 'http://localhost/rest/annotate'
    log.getLogger().setLevel(log.INFO)
    
    def __init__(self, confidence = 0.1, filter = None, columns = None):
        self.confidence = confidence
        self.filter = filter
        self.columns = columns

    def filter_dataset(self, dataset):
        counter = 0
        filtered_data = pd.DataFrame(columns = self.columns)

        for index, row in dataset.iterrows():
            comments = row["parent_comment"] + " " + row["comment"]
            try:
                spotlight.annotate(self.spotlight_endpoint, comments,
                                   confidence = self.confidence, filters = self.filter)
            except:
                sys.exc_clear()
            else :
                counter += 1
                if (counter % 1000 == 0):
                    log.info("Collected {0} matching records".format(counter))
                    filtered_data = filtered_data.append({
                        'label': row['label'],
                        'comment': row['comment'],
                        'parent_comment': row['parent_comment']
                      },
                      ignore_index = True)
        return filtered_data


input = pd.read_csv("/home/shariq/MSc/Research/dataset/reddit/pol/train-balanced.csv", sep='\t',
                    names = ["label", "comment", "auth", "subreddit", "score", "ups", "downs", 
                             "date", "created_utc", "parent_comment"])     
        

    
politics_filter = {'types' : "DBpedia:Politician," +
                   "DBpedia:PoliticalParty," +
                   "DBpedia:OfficeHolder," +
                   "DBpedia:PoliticalParty," +
                   "DBpedia:GovernmentAgency"}


col_names=["label", "comment", "parent_comment"]

politcs_extract = TopicExtractor(filter=politics_filter, columns=col_names)

type(politcs_extract)

mydata = politcs_extract.filter_dataset(input)
