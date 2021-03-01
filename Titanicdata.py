# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:39:26 2021

@author: shra1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(r'E:\CLASSES\TASK-13\TASK-13\DATASET\train.csv')
df.head()
df.describe()
del df ['Name']
del df ['Ticket']
del df ['Fare']
del df ['Cabin']
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
le.fit_transform(df['Sex'])
df['Sex']=le.fit_transform(df['Sex'])
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values='np.nan',strategy='mean')
imputer= imputer.fit(df['Age'])
df['Age']=imputer.transform(df['Age'])
df[:,0]
df.head()
def getnumber(str):
    if str=='male':
        return 1
    else:
        return 2
df['gender']=df['Sex'].apply(getnumber)
del df['Sex']
df.rename(columns={'gender':'Sex'},inplace=True)
meanS=df[df.Survived==1].Age.mean()
meanS
df['age']=np.where(pd.isnull(df.Age) & df['Survived']==1,meanS,df['Age'])
del df['Age']
df.rename(columns={'age':'Age'},inplace = True)
meanNS=df[df.Survived==0].Age.mean()
meanNS
df.Age.fillna(meanNS,inplace=True)
df.isnull().sum()
SurvivedQ=df[df.Embarked=='Q'][df.Survived==1].shape[0]
SurvivedS=df[df.Embarked=='S'][df.Survived==1].shape[0]
SurvivedC=df[df.Embarked=='C'][df.Survived==1].shape[0]
print(SurvivedC)
print(SurvivedS)
print(SurvivedQ)
SurvivedQ=df[df.Embarked=='Q'][df.Survived==0].shape[0]
SurvivedS=df[df.Embarked=='S'][df.Survived==0].shape[0]
SurvivedC=df[df.Embarked=='C'][df.Survived==0].shape[0]
print(SurvivedC)
print(SurvivedS)
print(SurvivedQ)
df.dropna(inplace=True)
df.isnull().sum()
def getEmb(str):
    if str=='S':
        return 1
    elif str=='C':
        return 2
    else:
        return 3
df['embark']=df['Embarked'].apply(getEmb)
del df['Embarked']
df.rename(columns={'embark':'Embarked'},inplace = True)
from matplotlib import style
male = (df['Sex']==1).sum()
female =(df['Sex']==0).sum()
p=[male,female]
plt.pie(p,    #giving array
       labels = ['Male', 'Female'], #Correspndingly giving labels
       colors = ['green','yellow'],   # Corresponding colors
       explode = (0.15, 0),    #How much the gap should me there between the pies
       startangle = 0)  #what start angle should be given
plt.show()
MaleS=df[df.Sex==1][df.Survived==1].shape[0]
print(MaleS)
MaleN=df[df.Sex==1][df.Survived==0].shape[0]
print(MaleN)
FemaleS=df[df.Sex==2][df.Survived==1].shape[0]
print(FemaleS)
FemaleN=df[df.Sex==2][df.Survived==0].shape[0]
print(FemaleN)
chart=[MaleS,MaleN,FemaleS,FemaleN]
colors=['lightskyblue','yellowgreen','Yellow','Orange']
labels=["Survived Male","Not Survived Male","Survived Female","Not Survived Female"]
explode=[0,0.05,0,0.1]
plt.pie(chart,labels=labels,colors=colors,explode=explode,startangle=100,counterclock=False,autopct="%.2f%%")
plt.axis("equal")
plt.show()