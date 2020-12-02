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
from pyod.models.sod import SOD
import preprocess
import pickle
from pyod.models.vae import VAE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


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

def model_test(model_type,y_train,y_test,X_train,X_test,model_file,save_flag):
    if model_type=='KNN':
        clf_name = 'KNN'
        clf = KNN()
        clf.fit(X_train)
    if model_type=='XGBOD':
        clf_name = 'XGBOD'
        clf = XGBOD(random_state=42)
        clf.fit(X_train, y_train)
    if model_type=='SOD':
        # train SOD detector
        # Note that SOD is meant to work in high dimensions d > 2.
        # But here we are using 2D for visualization purpose
        # thus, higher precision is expected in higher dimensions
        clf_name = 'SOD'
        clf = SOD()
        clf.fit(X_train)
    if model_type=='VAE':
        # train VAE detector (Beta-VAE)
        clf_name = 'VAE'
        contamination=0.01
        clf = VAE(epochs=30, contamination=contamination, gamma=0.8, capacity=0.2)
        clf.fit(X_train)

    #save model if specified
    if save_flag=='1':
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
    # visualize the results
    #todo： Input data has to be 2-d for visualization.
    #visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
     #         y_test_pred, show_figure=True, save_figure=False)

    return model_file

def main(input_file,file_type, label_col,model_file):
    if file_type=='file':
        print ("<<<<<< preprocess data")
        df = preprocess.main(input_file)
    if file_type=='folder':
        print("<<<<<< preprocess data")
        df = preprocess.process_file_list(input_file)
    print ("<<<<< data split")
    y_train,y_test,X_train,X_test = split_data(df,label_col)
    #data normalization
    mm = MinMaxScaler()
    X_train_std = mm.fit_transform(X_train)
    X_test_std = mm.fit_transform(X_test)

    for model_name in ['KNN','XGBOD','SOD']:
        print ("<<<<< model: ", model_name)
        model_test(model_name,y_train,y_test,X_train_std,X_test_std,model_file,'0')

if __name__ == "__main__":
    input_file= '~/Documents/zhongji/qingdao/data_IOT.xlsx'
    folder = ''
    model_file='./anno.model'
    #数据预处理
    input_folder = '../data'
    file_type = 'folder'
    main(input_folder,file_type,'Quality_modify',model_file)

