#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:42:52 2018

@author: shariq
"""

import TopicExtractor
import pandas as pd


class Main:
    
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
