# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:18:54 2021

@author: shra1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv(r'C:\Users\shra1\Downloads\KNN_Data.csv')

df.head()

sns.pairplot(data=df,hue='TARGET CLASS',palette='coolwarm')

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(df.drop('TARGET CLASS',axis=1))
SF=sc.transform(df.drop('TARGET CLASS',axis=1))

df_feat = pd.DataFrame(SF,columns=df.columns[:-1])
df_feat.head()

X= df_feat
Y=df['TARGET CLASS']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size =0.30,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,Y_train)

Y_pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_test, Y_pred))

error_rate = []

for i in range(1,40):
    knn= KNeighborsClassifier(n_neighbors= i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i !=Y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color ='blue',linestyle='--',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error rate vs K')
plt.xlabel('K')
plt.ylabel('Error rate')

knn = KNeighborsClassifier(n_neighbors = 40)
knn.fit(X_train,Y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_test, Y_pred))
