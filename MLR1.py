# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 15:56:55 2021

@author: shra1
"""

import pandas as pd 
import numpy as np
import seaborn as sns
df2=pd.read_csv(r'E:\CLASSES\15.6th Feb_Class15\15.6th Jan_Class15\TASK 12 -  TASK 17\TASK-17\kc_house_data.csv')
df2.head()
del df2['id']
del df2['date']
print(df2.dtypes)
X = df2.iloc[:,1:].values
y = df2.iloc[:,0].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
Regressor=LinearRegression()
Regressor.fit(X_train,y_train)

y_pred = Regressor.predict(X_test)

import statsmodels.formula.api as sm
df2
X= np.append(arr= np.ones((21613,1).astype(int),values= X,axis=1)
X = np.append(arr = np.ones((21613,1)).astype(int), values = X, axis = 1)
import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS =sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
