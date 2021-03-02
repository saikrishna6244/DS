# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:56:41 2021

@author: shra1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 df = pd.read_csv(r'C:\Users\shra1\Downloads\archive\advertising.csv')

df.head()
df.info()
df.describe()

df['Age'].plot.hist(bins=35,color='red')

sns.jointplot(x='Area Income',y='Age',data=df)

sns.jointplot(x='Age', y='Daily Internet Usage',data= df, kind = "kde",color= 'r')

sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage',data=df)

sns.pairplot(df,hue='Clicked on Ad')
df=df.drop(['Ad Topic Line','City','Country','Timestamp'], axis =1)
from sklearn.model_selection import train_test_split
X= df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.30,random_state = 0)

X['']

from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_test,Y_pred))
print(confusion_matrix(Y_test, Y_pred))
