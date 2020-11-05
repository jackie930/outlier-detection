# -*- coding: utf-8 -*-
# @Time    : 11/5/20 3:06 PM
# @Author  : Jackie
# @File    : forecast.py.py
# @Software: PyCharm
import preprocess
import os
import pickle
import numpy as np

class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            with open(os.path.join('./anno.model'), 'rb') as f:
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

# ## last data
def infer_anomaly_model(clf, infer_data):
    print ("<<<<<< preprocess data")
    df = preprocess.main(infer_data)

    X_test = np.array(df.iloc[:,1:])
    X_test = X_test.astype('int')

    y_test_pred = clf.predict(X_test)
    print (y_test_pred)
    return y_test_pred


def main():
    global clf
    clf = ScoringService.get_model()
    infer_anomaly_model(clf, '~/Documents/zhongji/qingdao/data_IOT.xlsx')

if __name__ == "__main__":
    main()