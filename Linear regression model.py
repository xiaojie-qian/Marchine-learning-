#!/usr/bin/env python
# coding: utf-8

# # Week 2

# ## 1. Multiple linear regression

# In[1]:


import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'
df = pd.read_csv(url)


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df.describe()


# In[6]:


df.dtypes


# In[7]:


df.corr()


# ### Split the training and test data

# #### *To split the dataset using numpy*

# In[9]:


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test =cdf[~msk]


# ## 2. Simple regresson model

# In[10]:


plt.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'] ,color ='blue')
plt.xlabel('enginesize')
plt.ylabel('Co2 emissions')
plt.show()


# In[11]:


from sklearn import linear_model


# In[12]:


from sklearn.linear_model import LinearRegression


# In[21]:


regr = LinearRegression()
regr


# In[22]:


# train_x = np.asanyarray(train[['ENGINESIZE']])
# train_y = np.asanyarray(train[['CO2EMISSIONS']])
# Use asanyarray to turn from 1D array to nD array

regr.fit(train[['ENGINESIZE']], train['CO2EMISSIONS'])
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[27]:


yhat_e = regr.predict(train[['ENGINESIZE']])


# In[28]:


sns.regplot(x=train[['ENGINESIZE']],y=train['CO2EMISSIONS'])


# In[40]:


ax1 = sns.kdeplot(train['CO2EMISSIONS'], color = 'r', label = 'acutal values')
sns.kdeplot(yhat_e, color = 'b', label = 'fitted values', ax = ax1)
ax1.set_title('actual vs. fitted values of enginesize')
ax1.legend()
plt.show()


# In[42]:


regr.score(test[['ENGINESIZE']], test['CO2EMISSIONS'])


# In[51]:


from sklearn.metrics import mean_squared_error


# In[55]:


mse_s = mean_squared_error(train['CO2EMISSIONS'], yhat_e)
mse_s


# ## 3. Multiple regresson model

# In[8]:


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
plt.scatter(cdf['ENGINESIZE'], cdf['CO2EMISSIONS'] , color = 'green')
plt.xlabel('enginesize')
plt.ylabel('Co2 emissions')
plt.show()


# In[56]:


plt.scatter(cdf['CYLINDERS'], cdf['CO2EMISSIONS'] , color = 'green')
plt.xlabel('cylinders')
plt.ylabel('Co2 emissions')
plt.show()


# In[61]:


plt.scatter(cdf['FUELCONSUMPTION_COMB'], cdf['CO2EMISSIONS'] , color = 'green')
plt.xlabel('fuel-consumption_comb')
plt.ylabel('Co2 emissions')
plt.show()


# In[67]:


x_train= train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
y_train = train['CO2EMISSIONS']
regr.fit(x_train,y_train)


# In[68]:


yhat_m = regr.predict(x_train)
yhat_m[0:4]


# #### Ordinary Least Squares (OLS)
# 
# OLS is a method for estimating the unknown parameters in a linear regression model. OLS chooses the parameters of a linear function of a set of explanatory variables by minimizing the sum of the squares of the differences between the target dependent variable and those predicted by the linear function. In other words, it tries to minimizes the sum of squared errors (SSE) or mean squared error (MSE) between the target variable (y) and our predicted output ($\hat{y}$) over all samples in the dataset.
# 
# OLS can find the best parameters using of the following methods:
# 
# *   Solving the model parameters analytically using closed-form equations
# *   Using an optimization algorithm (Gradient Descent, Stochastic Gradient Descent, Newton’s Method, etc.)

# In[69]:


mse = mean_squared_error(y_train,yhat_m)
mse


# In[70]:


ax1 = sns.kdeplot(y_train, color = 'r', label = 'acutal values')
sns.kdeplot(yhat_m, color = 'b', label = 'fitted values', ax = ax1)
ax1.set_title('actual vs. fitted values of multiple variables')
ax1.legend()
plt.show()


# In[71]:


x_test= test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
y_test = test[['CO2EMISSIONS']]
regr.score(x_test,y_test)


# ## 3. Polynomial Regression

# **PolynomialFeatures()** function in Scikit-learn library, drives a new feature sets from the original feature set. That is, a matrix will be generated consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, lets say the original feature set has only one feature, *ENGINESIZE*. Now, if we select the degree of the polynomial to be 2, then it generates 3 features, degree=0, degree=1 and degree=2:
# 

# In[85]:


from sklearn.preprocessing import PolynomialFeatures


# In[116]:


poly = PolynomialFeatures(degree = 2)
x_train_pr = poly.fit_transform(train[['ENGINESIZE']])
x_test_pr = poly.fit_transform(test[['ENGINESIZE']])
poly


# In[117]:


lm_pr = LinearRegression()


# In[118]:


y_train_pr = train[['CO2EMISSIONS']]
y_test_pr = test['CO2EMISSIONS']


# In[119]:


lm_pr.fit(x_train_pr,y_train_pr)


# In[120]:


yhat_pr = lm_pr.predict(x_train_pr)
yhat_pr[0:4]


# In[121]:


mse_pr = mean_squared_error(y_train_pr,yhat_pr)
mse_pr


# In[122]:


sns.regplot(x = train[['ENGINESIZE']], y= train[['CO2EMISSIONS']])


# ### visulize a polynomial regression

# In[108]:


ax1 = sns.kdeplot(y_train_pr, color = 'r', label = 'acutal values')
sns.kdeplot(yhat_pr, color = 'b', label = 'fitted values', ax = ax1)
ax1.set_title('actual vs. fitted values of multiple variables')
ax1.legend()
plt.show()


# In[109]:


lm_pr.score(x_test_pr,y_test_pr)


# In[ ]:




