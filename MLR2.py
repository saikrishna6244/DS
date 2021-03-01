# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 22:29:16 2021

@author: shra1
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

df=pd.read_csv(r'E:\CLASSES\15.6th Feb_Class15\15.6th Jan_Class15\TASK 12 -  TASK 17\TASK-15\50_Startups.csv')
df.head()
df['State'].unique()
df=pd.get_dummies(data=df,columns=['State'])
df.info()
df.shape
Y=df['Profit']
X=df.drop(columns='Profit',axis=1)
from sklearn.model_selection import train_test_split
type(X)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
Y_pred=lr.predict(X_test)
import statsmodels.formula.api as sm
df
X= np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
import statsmodels.api as sm
X_opt=X[:,[0,1,2,3,4,5]]
lr_ols=sm.OLS(endog=Y,exog=X_opt).fit()
lr_ols.summary()
X_opt = X[:, [0,1,2,3,5]]
lr_ols=sm.OLS(endog=Y,exog=X_opt).fit()
lr_ols.summary()
X_opt =X[:,[0,1,2,3]]
lr_ols=sm.OLS(endog=Y,exog=X_opt).fit()
lr_ols.summary()
X_opt =X[:,[0,1,3]]
lr_ols=sm.OLS(endog=Y,exog=X_opt).fit()
lr_ols.summary()
X_opt =X[:,[0,1]]
lr_ols=sm.OLS(endog=Y,exog=X_opt).fit()
lr_ols.summary()
