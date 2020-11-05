# -*- coding: utf-8 -*-
# @Time    : 10/21/20 4:28 PM
# @Author  : Jackie
# @File    : model.py
# @Software: PyCharm
from sklearn.model_selection import train_test_split
import numpy as np
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from pyod.models.xgbod import XGBOD
import preprocess
import pickle


def split_data(input1,label_col):
    df_train,df_test = train_test_split(input1,test_size = 0.2,stratify=input1[label_col])
    X_train = np.array(df_train.iloc[:,1:])
    y_train = np.array(df_train.iloc[:,0])
    X_test = np.array(df_test.iloc[:,1:])
    y_test = np.array(df_test.iloc[:,0])

    # type conversion
    y_test = y_test.astype('int')
    X_test = X_test.astype('int')
    return y_train,y_test,X_train,X_test

def model(model_type,y_train,y_test,X_train,X_test,model_file):
    if model_type=='KNN':
        clf_name = 'KNN'
        clf = KNN()
        clf.fit(X_train)
        #save model file
        pickle.dump(clf, open(model_file, "wb"))
    if model_type=='XGBOD':
        clf_name = 'XGBOD'
        clf = XGBOD(random_state=42)
        clf.fit(X_train, y_train)
        # save model to file
        pickle.dump(clf, open(model_file, "wb"))

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)

    return model_file

def main(input_file,label_col,model_file):
    print ("<<<<<< preprocess data")
    df = preprocess.main(input_file)
    print ("<<<<< data split")
    y_train,y_test,X_train,X_test = split_data(df,label_col)
    print ("<<<<< models KNN")
    model("KNN",y_train,y_test,X_train,X_test,model_file)
    print ("<<<<< models xgbod")
    model("XGBOD",y_train,y_test,X_train,X_test,model_file)

if __name__ == "__main__":
    input_file= '~/Documents/zhongji/qingdao/data_IOT.xlsx'
    folder = ''
    model_file='./anno.model'
    #数据预处理
    main(input_file,'Quality_modify',model_file)

