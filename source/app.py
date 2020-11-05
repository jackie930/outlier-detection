# -*- coding: utf-8 -*-
# @Time    : 11/5/20 3:11 PM
# @Author  : Jackie
# @File    : app.py.py
# @Software: PyCharm

import os
import pickle
import io
import flask
import pandas as pd
import json

from forecast import infer_anomaly_model

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            with open('./anno.model', 'rb') as f:
                cls.model = pickle.load(f)
        return cls.model

    @classmethod
    def predict(cls, x):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf.predict(x)

# The flask app for serving predictions
app = flask.Flask(__name__)
clf = None  # You can insert a health check here

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    global clf
    if clf is None:
        clf = ScoringService.get_model()
    status = 200
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/init', methods=['GET'])
def init_mode():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    global clf
    if clf is None:
        clf = ScoringService.get_model()
    status = 200
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    if flask.request.content_type == 'application/json':
        input_string = flask.request.data.decode('utf-8')
        print(input_string)
        a = json.loads(input_string)
        print(type(a))
        x = pd.DataFrame([a])
        print(type(x))
        print(x.head())
        result = infer_anomaly_model(clf,x)
        x["class"]=result
        #x["class"]=0
        response=x.to_json(orient="records",force_ascii=False)
        return flask.Response(response=response, status=200, mimetype='application/json')
    else:
        return flask.Response(response='This predictor only supports Json data', status=415, mimetype='text/plain')