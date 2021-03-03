# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:39:37 2021

@author: shra1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

cancer.keys()

print(cancer['data'])

print(cancer['feature_names'])
cancer['data'].shape

df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns= np.append(cancer['feature_names'],['target']))

sns.pairplot(df_cancer,vars =['mean radius','mean texture','mean perimeter','mean area',
 'mean smoothness'])

sns.pairplot(df_cancer,hue='target',vars =['mean radius','mean texture','mean perimeter','mean area'])

sns.countplot(df_cancer['target'])

sns.scatterplot(x='mean area',y ='mean smoothness',hue = 'target',data = df_cancer)
plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(),annot= True)

X = df_cancer.drop(['target'],axis = 1)
Y = df_cancer['target']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.20, random_state=5)

from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

svc_mdl = SVC()

svc_mdl.fit(X_train,Y_train)

Y_pred =svc_mdl.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm,annot=True)

min_train = X_train.min()
range_train= (X_train - min_train).max()
X_train_scaled = (X_train -min_train)/range_train

min_test = X_test.min()
range_test= (X_test - min_test).max()
X_test_scaled = (X_test -min_test)/range_test

svc_mdl.fit(X_train_scaled,Y_train)
Y_pred =svc_mdl.predict(X_test_scaled)
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm,annot=True)
cr=classification_report(Y_test, Y_pred)
