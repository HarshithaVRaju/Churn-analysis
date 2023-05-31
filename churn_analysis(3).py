# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 09:28:02 2022

@author: Harshitha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import os
os.getcwd()
os.chdir("C:/Users/Harshitha/OneDrive/Documents/python code")
mydata=pd.read_csv("Telecom_Data.csv")
print(mydata.head())
print(mydata.isnull().sum())
print(mydata.dtypes)
print(mydata.describe(include=["object"]))
y=mydata[["churn"]]
#x=mydata[["state","account length","area code","phone number","international plan","voice mail plan",
        #"number vmail messages","total day minutes ","total day calls","total day charge",
      #  "total eve minutes","total eve calls",]]
x=mydata.drop(["churn"],axis=1)
#extremeboost algo is used from xgboost,bagging predicts parallely all outputs while boosting does it sequentially
x=pd.get_dummies(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20)

from xgboost import XGBClassifier
model=XGBClassifier()
model.fit(xtrain,ytrain)
p_value=model.predict(xtest)
confusion_matrix(ytest, p_value)
from sklearn.metrics import classification_report
classification_report(ytest,p_value)


