#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd

import csv

#read csv file
df = pd.read_csv(r"C:\Users\Francis Apolinar\Desktop\Emotion_Dataset\Emotion_final.csv")


# In[11]:


#check file if working
df.head()


# In[6]:


#print all lines from csv file
#print(df)

#collate data of the csv file
df.describe()


# In[13]:


#count amount of emotions in file
#0 - anger, 1 - fear, 2 - Happy, 3 - love, 4 - sadness, 5 - surprise
df['Emotion'].value_counts()


# In[7]:


#Extracting and separating data based on the labels from the data set
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
df['Emotion'] = label_encoder.fit_transform(df['Emotion'])
train = df


# In[9]:


#Data cleaning

#removing hastags
train['Text'].replace({ r"#(\w+)" : '' }, inplace = True, regex  =True)

#removing mentions
train['Text'].replace({ r"@(\w+)" : '' }, inplace = True, regex  =True)

#removing URL
train['Text'].astype(str).replace({ r"http\S+" : '' }, inplace = True, regex  =True)


# In[10]:


#converting text into small case
train['Text'] = train['Text'].str.lower()
df_train = train


# In[ ]:




