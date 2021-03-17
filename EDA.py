# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:12:24 2021

@author: shra1
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')

df =  pd.read_csv(r'E:\CLASSES\3. AMXWAM_ TASK\3. AMXWAM_ TASK\TASK-39 to TASK-49\heart-disease-uci_ TASK 42\heart.csv')
df.head()

df.isnull().sum()
df.isna().sum()
df.describe(include='all')
df.columns
df['target'].nunique()
df['target'].value_counts()

f, ax = plt.subplots(figsize=(6, 6))
ax = sns.countplot(x="target", data=df)
plt.show()

df.groupby('sex')['target'].value_counts()

f,ax = plt.subplots(figsize=(8,6))
ax = sns.countplot(x="sex",hue="target",data=df)
plt.show()

ax = sns.catplot(x="sex",hue="target",data=df,kind="count",color='red')

f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target", data=df, palette="Set3")
plt.show()

f,ax =plt.subplots(figsize=(8,6))
ax= sns.countplot(x="target",hue="fbs",data=df)
plt.show()

correlation = df.corr()

correlation['target'].sort_values(ascending=False)

df['cp'].nunique()

df['cp'].value_counts()

f,ax = plt.subplots(figsize=(8,6))
x = df['thalach']
ax = sns.distplot(x,bins=10)
plt.show()

f, ax = plt.subplots(figsize=(10,6))
x = df['thalach']
x = pd.Series(x, name="thalach variable")
ax = sns.distplot(x, bins=10)
plt.show()

f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="thalach", data=df, jitter = 0.01)
plt.show()

plt.figure(figsize=(16,12))
plt.title('Correlation Heatmap of Heart Disease Dataset')
a = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=30)           
plt.show()

assert pd.notnull(df).all().all()
