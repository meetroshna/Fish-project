#!/usr/bin/env python
# coding: utf-8

# In[10]:


#load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[11]:


#load dataset
df=pd.read_csv(r"C:\Users\roshn\OneDrive\Desktop\Fish.csv")


# In[12]:


df.head()


# In[13]:


df.info()


# In[14]:


df.describe()


# In[15]:


df.corr()


# In[16]:


df.isnull().sum()


# In[19]:


X = df.drop('Species', axis=1)
y = df['Species']


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25, random_state=42)

from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=6)
selector.fit(X_train, y_train)


# In[21]:


selector.scores_


# In[22]:


cols = selector.get_support(indices=True)
cols


# In[23]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
#encoder = LabelEncoder()
#y = encoder.fit_transform(y)
#y_map = {index:label for index,label in enumerate(encoder.classes_)}


# In[24]:


#y_map


# In[25]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[26]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[27]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25, random_state=42)


# In[31]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train,y_train)


# In[32]:


log_model.score(X_test,y_test)


# In[35]:


# Create a Pickle file  
import pickle
pickle_out = open("fish.pkl","wb")
pickle.dump(log_model, pickle_out)
pickle_out.close()


# In[ ]:




