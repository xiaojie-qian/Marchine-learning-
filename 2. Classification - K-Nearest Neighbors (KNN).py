#!/usr/bin/env python
# coding: utf-8

# ## Classification 

# ### 1. pre-process the data

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"
df = pd.read_csv(url)


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df.dtypes


# In[7]:


df.describe()


# In[8]:


df['custcat'].value_counts()


# In[20]:


df.hist(column = ['age', 'income'], bins = 30)


# In[21]:


x = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
       'employ', 'retire', 'gender', 'reside']]
x.head()


# In[23]:


y = df['custcat'].values
y[0:4]


# ### 2. Normalize data

# In[31]:


from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[34]:


# data standardization 
scaler = StandardScaler()
x_norm = scaler.fit_transform(x.astype(float))
x_norm[0:4]


# ### 3. Train test split

# In[49]:


x_train,x_test,y_train,y_test = train_test_split(x_norm, y, test_size = 0.2, random_state = 0)
print('number of train samples: ', x_train.shape[0])
print('number of test samples: ', len(x_test))


# ### 4. K_nearest neighbors

# In[50]:


from sklearn.neighbors import KNeighborsClassifier


# In[51]:


k = 4
#  train model and predict 
neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
neigh


# In[52]:


yhat_k4 = neigh.predict(x_test)
yhat_k4[0:4]


# #### Accuracy evaluation

# In[53]:


from sklearn import metrics


# In[57]:


# jaccard_score function
train_score = metrics.accuracy_score(y_train, neigh.predict(x_train))
test_score = metrics.accuracy_score(y_test, yhat_k4)
print('Train set score: ', train_score)
print('Test set score:', test_score)


# In[87]:


ks = 10

for n in range (1,ks):
    neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    yhat = neigh.predict(x_test)
    test_score = metrics.accuracy_score(y_test, yhat)
    print('k = ', n)
    print('test_score=', test_score)


# In[91]:


# calculate the accuracy of KNN

ks = 10
mean_acc = np.zeros((ks-1))
std_acc = np.zeros((ks-1))

for n in range (1,ks):
    neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    yhat = neigh.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    print('k = ', n)
    print('test_score=', mean_acc[n-1])
    


# In[73]:


k = 5
neigh_k5 = KNeighborsClassifier(n_neighbors = 5).fit(x_train,y_train)
neigh_k5


# In[74]:


# train accuracy score when k = 5
metrics.accuracy_score(y_train,neigh_k5.predict(x_train))


# In[75]:


# test accuracy score when k = 5
metrics.accuracy_score(y_test,neigh_k5.predict(x_test))


# In[92]:


#plot the model accuracy for a different number of neighbors
ks = 10


plt.plot(range(1,ks),mean_acc,'g')
plt.fill_between(range(1,ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# In[ ]:




