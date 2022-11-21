#!/usr/bin/env python
# coding: utf-8

# # SVM (Support Vector Machines)
# 
# ## Objectives
# 
# After completing this lab you will be able to:
# 
# *   Use scikit-learn to Support Vector Machine to classify
# 
# SVM works by mapping data to a high-dimensional feature space so that data points can be categorized, even when the data are not otherwise linearly separable. A separator between the categories is found, then the data is transformed in such a way that the separator could be drawn as a hyperplane. Following this, characteristics of new data can be used to predict the group to which a new record should belong.
# 
# | Field name  | Description                 |
# | ----------- | --------------------------- |
# | ID          | Clump thickness             |
# | Clump       | Clump thickness             |
# | UnifSize    | Uniformity of cell size     |
# | UnifShape   | Uniformity of cell shape    |
# | MargAdh     | Marginal adhesion           |
# | SingEpiSize | Single epithelial cell size |
# | BareNuc     | Bare nuclei                 |
# | BlandChrom  | Bland chromatin             |
# | NormNucl    | Normal nucleoli             |
# | Mit         | Mitoses                     |
# | Class       | Benign or malignant         |

# ## 1. Read the data

# In[109]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[128]:


url ="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv"
df = pd.read_csv(url)
df.head()


# In[129]:


df.shape


# In[130]:


df.columns


# In[131]:


df.dtypes


# In[132]:


df.describe()


# In[133]:


df.isnull().sum()


# ## 2. pre-process

# In[164]:


#Let's look at the distribution of the classes based on Clump thickness and Uniformity of cell size:
df['Clump'].unique()


# In[165]:


df['UnifSize'].unique()


# In[166]:


df['Class'].unique()


# In[167]:


ax = df[df['Class'] == 4][0:99].plot(kind='scatter', x='Clump', y='UnifSize', color='red', label='malignant')
ax1 = df[df['Class'] == 2][0:99].plot(kind='scatter', x='Clump', y = 'UnifSize', color = 'yellow', label = 'Benign', ax = ax)
plt.show()


# In[168]:


df['BareNuc'].unique()


# In[169]:


len(df['BareNuc'])


# In[170]:


df[df["BareNuc"].str.contains('4')].count()


# In[171]:


# filter values contain '?'
cell_df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]


# In[172]:


cell_df.dtypes


# In[173]:


cell_df['BareNuc'].unique()


# In[174]:


cell_df['BareNuc'].astype(int)


# ### Divide predictors and response 

# In[212]:


x = cell_df.drop(columns = ['ID','Class']).values
x[0:4]


# In[213]:


y = cell_df['Class']
y[0:5]


# In[249]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
print('train size: ', x_train.shape[0])
print('test size: ', x_test.shape[0])


# ## 3. Modeling and evaluation 

# The SVM algorithm offers a choice of kernel functions for performing its processing. Basically, mapping data into a higher dimensional space is called kernelling. The mathematical function used for the transformation is known as the kernel function, and can be of different types, such as:
# 
# ```
# 1.Linear
# 2.Polynomial
# 3.Radial basis function (RBF)
# 4.Sigmoid
# ```
# 
# Each of these functions has its characteristics, its pros and cons, and its equation, but as there's no easy way of knowing which function performs best with any given dataset. We usually choose different functions in turn and compare the results. Let's just use the default, RBF (Radial Basis Function) for this lab.

# In[250]:


from sklearn import svm
svm = svm.SVC(kernel = 'rbf')
# Support vector classifier
svm.fit(x_train, y_train)


# In[251]:


yhat = svm.predict(x_test)
yhat[0:50]


# In[252]:


from sklearn.metrics import confusion_matrix
svm_matrix = confusion_matrix(y_test,yhat, labels = [2,4])
svm_matrix


# In[253]:


import seaborn as sns

text = np.array([['TP', 'FN'], ['FP','TN']])

# combining text with values
formatted_text = (np.array(["{0}\n{1:0}".format(
    text, svm_matrix) for text, svm_matrix in zip(text.flatten(), svm_matrix.flatten())])).reshape(2, 2)
# Reshape data array and text array into 1D using np.flatten().
# Then zip them together to iterate over both text and value.
# Use formatted strings to create customized new value.
# Return a reshaped array of the same size containing customized values.

ax = sns.heatmap(svm_matrix, annot = formatted_text, fmt = "", cmap='Oranges')

ax.set_title('Confusion matrix');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Benign = 2','Malignent = 4'])
ax.yaxis.set_ticklabels(['Benign = 2','Malignent = 4'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[254]:


from sklearn.metrics import classification_report
print(classification_report(y_test,yhat))


# In[255]:


print('precision % .2f '%(83/(83+1)))


# In[256]:


print('precision % .2f '%(83/(83+4)))


# In[257]:


from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 


# In[ ]:


"""from sklearn.svm import SVC
svm2 = svm.SVC(kernel='linear')
svm2.fit(X_train, y_train) 
yhat2 = svm2.predict(X_test)
print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
print("Jaccard score: %.4f" % jaccard_score(y_test, yhat2,pos_label=2))"""


# In[ ]:




