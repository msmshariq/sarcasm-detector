#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: shariq
"""

from flask import Flask, render_template, request
app = Flask(__name__)
 

@app.route('/')
def showSignUp():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def signUp():
    print('In backend')
    parentComment = request.form['parentComment']
    comment = request.form['comment']
    print(parentComment)
    return parentComment + "-" + comment 
 
if __name__ == "__main__":
    app.run()