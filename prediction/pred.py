import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
#from datetime import datetime
import datetime
import os

script_dir = os.path.dirname(__file__)

def pred_decision(dataset):
	#dataset = dataset1
    X = dataset.iloc[:,[2,3]].values
    y = dataset.iloc[:,4].values


    #splitting the dataset into the training set and test set (train split -function)
    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.25, random_state = 0)

    #X_train
    #X_test
    #y_train
    #y_test

    #Feature scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    #FITTING DECISION TREE CLASSIFICATION TO THE TRAINING SET
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
    #classifier is the machine which is going to learn 
    classifier.fit(X_train,y_train)

    y_pred = classifier.predict(X_test)

    '''with open(os.path.join(dir_path,id + ".log"), 'w') as file:
        for line in y_pred:
            file.write(str(line))'''
    return y_pred



