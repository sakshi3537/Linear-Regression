#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importing the dataset
companies=pd.read_csv(r'C:\Users\HP\Downloads\Machine Learning Full\Linear Regression\1000_Companies.csv')
X=companies.iloc[:,:-1].values
y=companies.iloc[:,4].values


# In[3]:


companies.head()


# In[4]:


sns.heatmap(companies.corr())


# In[6]:


#For categorical variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])

onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

X=X[:,1:]


# In[7]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[8]:


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
Regressor=LinearRegression()
Regressor.fit(X_train,y_train)


# In[9]:


# Predicting the Test set results
y_pred=Regressor.predict(X_test)
print(y_pred)


# In[10]:


#Overview of slope and intercept
print(Regressor.coef_)
print(Regressor.intercept_)


# In[11]:


#Accuracy of model
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[ ]:




