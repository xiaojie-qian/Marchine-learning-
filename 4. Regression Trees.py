#!/usr/bin/env python
# coding: utf-8

# ## Objectives
# *   Train a Regression Tree
# *   Evaluate a Regression Trees Performance

# ## 1. Read the data

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv"
df = pd.read_csv(url)
df.head()


# #### About the data
# A real estate company is planning to invest in Boston real estate. You have collected information about various areas of Boston and are tasked with created a model that can predict the median price of houses for that area so it can be used to make offers.
# 
# The dataset had information on areas/towns not individual houses, the features are
# 
# CRIM: Crime per capita
# 
# ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.
# 
# INDUS: Proportion of non-retail business acres per town
# 
# CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 
# NOX: Nitric oxides concentration (parts per 10 million)
# 
# RM: Average number of rooms per dwelling
# 
# AGE: Proportion of owner-occupied units built prior to 1940
# 
# DIS: Weighted distances to ﬁve Boston employment centers
# 
# RAD: Index of accessibility to radial highways
# 
# TAX: Full-value property-tax rate per $10,000
# 
# PTRAIO: Pupil-teacher ratio by town
# 
# LSTAT: Percent lower status of the population
# 
# MEDV: Median value of owner-occupied homes in $1000s

# In[6]:


df.dtypes


# In[7]:


df.columns


# In[9]:


df.shape


# In[11]:


df.describe()


# In[19]:


df.isna().sum()


# ## 2.Pre-processing

# In[21]:


df.dropna(inplace = True)


# In[22]:


df.isna().sum()


# In[26]:


x = df.drop(columns = ['MEDV']).values.astype(float)
x[0:4]


# In[25]:


y = df['MEDV']
y.head()


# ## 3. Modeling 

# In[29]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[31]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state = 0)
print('train size: ', x_train.shape[0])
print('test size: ', x_test.shape[0])


# In[35]:


rt = DecisionTreeRegressor(criterion = 'squared_error')
rt.fit(x_train,y_train)


# In[40]:


print('accuracy score= %.2f'% rt.score(x_test,y_test))


# In[42]:


yhat = rt.predict(x_test)
yhat[0:4]


# In[45]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,yhat)
mse


# In[58]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, yhat)
print('$%.3f'% (mae * 1000))


# In[53]:


# change criterion to mae
rt_2 = DecisionTreeRegressor(criterion = 'absolute_error')
rt_2.fit(x_train, y_train)
print('accuracy score: %.2f'% rt_2.score(x_test,y_test))


# In[59]:


yhat_2 = rt_2.predict(x_test)
mae = mean_absolute_error(y_test,yhat_2)
print('$%.3f'%(mae*1000))


# In[ ]:




