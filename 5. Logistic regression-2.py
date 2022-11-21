#!/usr/bin/env python
# coding: utf-8

# <a id="ref1"></a>
# 
# ## What is the difference between Linear and Logistic Regression?
# 
# While Linear Regression is suited for estimating continuous values (e.g. estimating house price), it is not the best tool for predicting the class of an observed data point. In order to estimate the class of a data point, we need some sort of guidance on what would be the <b>most probable class</b> for that data point. For this, we use <b>Logistic Regression</b>.
# 
# <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# <font size = 3><strong>Recall linear regression:</strong></font>
# <br>
# <br>
#     As you know, <b>Linear regression</b> finds a function that relates a continuous dependent variable, <b>y</b>, to some predictors (independent variables $x_1$, $x_2$, etc.). For example, simple linear regression assumes a function of the form:
# <br><br>
# $$
# y = \theta_0 + \theta_1  x_1 + \theta_2  x_2 + \cdots
# $$
# <br>
# and finds the values of parameters $\theta_0, \theta_1, \theta_2$, etc, where the term $\theta_0$ is the "intercept". It can be generally shown as:
# <br><br>
# $$
# ℎ_\theta(𝑥) = \theta^TX
# $$
# <p></p>
# 
# </div>
# 
# Logistic Regression is a variation of Linear Regression, used when the observed dependent variable, <b>y</b>, is categorical. It produces a formula that predicts the probability of the class label as a function of the independent variables.
# 
# Logistic regression fits a special s-shaped curve by taking the linear regression function and transforming the numeric estimate into a probability with the following function, which is called the sigmoid function 𝜎:
# 
# $$
# ℎ\_\theta(𝑥) = \sigma({\theta^TX}) =  \frac {e^{(\theta\_0 + \theta\_1  x\_1 + \theta\_2  x\_2 +...)}}{1 + e^{(\theta\_0 + \theta\_1  x\_1 + \theta\_2  x\_2 +\cdots)}}
# $$
# Or:
# $$
# ProbabilityOfaClass\_1 =  P(Y=1|X) = \sigma({\theta^TX}) = \frac{e^{\theta^TX}}{1+e^{\theta^TX}}
# $$
# 
# In this equation, ${\theta^TX}$ is the regression result (the sum of the variables weighted by the coefficients), `exp` is the exponential function and $\sigma(\theta^TX)$ is the sigmoid or [logistic function](http://en.wikipedia.org/wiki/Logistic_function?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01), also called logistic curve. It is a common "S" shape (sigmoid curve).
# 
# So, briefly, Logistic Regression passes the input through the logistic/sigmoid but then treats the result as a probability:
# 
# <img
# src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/images/mod_ID_24_final.png" width="400" align="center">
# 
# The objective of the **Logistic Regression** algorithm, is to find the best parameters θ, for $ℎ\_\theta(𝑥)$ = $\sigma({\theta^TX})$, in such a way that the model best predicts the class of each case.
# 

# ## About the dataset : 
# ### Customer churn with Logistic Regression
# 
# A telecommunications company is concerned about the number of customers leaving their land-line business for cable competitors. They need to understand who is leaving. Imagine that you are an analyst at this company and you have to find out who is leaving and why.
# 
# We will use a telecommunications dataset for predicting customer churn. This is a historical customer dataset where each row represents one customer. The data is relatively easy to understand, and you may uncover insights you can use immediately. Typically it is less expensive to keep customers than acquire new ones, so the focus of this analysis is to predict the customers who will stay with the company. 
# 
# This data set provides information to help you predict what behavior will help you to retain customers. You can analyze all relevant customer data and develop focused customer retention programs.
# 
# The dataset includes information about:
# 
# *   Customers who left within the last month – the column is called Churn
# *   Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# *   Customer account information – how long they had been a customer, contract, payment method, paperless billing, monthly charges, and total charges
# *   Demographic info about customers – gender, age range, and if they have partners and dependents

# In[3]:


import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt


# In[4]:


url =  "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
df = pd.read_csv(url)
df.head()


# In[5]:


df.columns


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.dtypes


# In[9]:


missing_value = df.isnull()
missing_value[0:4]


# In[10]:


df['churn'].unique()


# ## Data pre-process

# In[11]:


x = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless']].values
x[0:4]


# In[12]:


y = df['churn'].astype('int')
y.head()


# In[13]:


# data standardization 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
x[0:4]


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
print('train size is: ',x_train.shape[0])
print('test size is: ', x_test.shape[0])


# ## Modeling 

# **LogisticRegression:** This function implements logistic regression and can use different numerical optimizers to find parameters, including ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers. You can find extensive information about the pros and cons of these optimizers if you search it in the internet.
# 
# The version of Logistic Regression in Scikit-learn, support regularization. Regularization is a technique used to solve the overfitting problem of machine learning models.
# **C** parameter indicates **inverse of regularization strength** which must be a positive float. Smaller values specify stronger regularization.
# Now let's fit our model with train set:

# In[16]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 0.01, solver = 'liblinear')
lr.fit(x_train,y_train)


# In[17]:


# This output gives the probabilities of the occurrence of each target every tuple of the testing set.
yhat_prob = lr.predict_proba(x_test)
yhat_prob


# In[18]:


yhat = lr.predict(x_test)
yhat
# predict_proba returns estimates for all classes, ordered by the label of classes. 


# In[55]:


np.array(y_test)


# **predict_proba**  returns estimates for all classes, ordered by the label of classes. So, the first column is the probability of class 0, P(Y=0|X), and second column is probability of class 1, P(Y=1|X):

# ### Evaluation: jaccard index
# we can define jaccard as the size of the intersection divided by the size of the union of the two label sets. If the entire set of predicted labels for a sample strictly matches with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.
# ![image.png](attachment:image.png)
# 

# In[19]:


from sklearn.metrics import jaccard_score
jaccard_score(y_test,yhat, pos_label = 0)
# The pos_label parameter lets you specify which class should be considered "positive"


# ### Confusion matrix

# In[20]:


"""from sklearn.metrics import confusion_matrix
import itertools"""


# In[40]:


"""def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    if normalize:
        cm = cm.astype(float)/cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else: 
        print('Confusion matrix, without normalization')
    print(cm)
    
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.sf' if normalize else 'd'
    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),
                horizontalalignment = 'center',
                color = 'white' if cm [i,j] > thresh else 'black')
        
    plt.tight_layout()
    plt.ylabel('True lable')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels = [1,0]))"""


# In[22]:


"""# compute confusio matrix
cnf_matrix = confusion_matrix(y_test,yhat, labels = [1,0])
np.set_printoptions(precision = 2)

# Plot non-normlized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = ['churn = 1', 'chrun = 0'], normalize = False, title = 'Cnofusion matrix')"""


# In[49]:


from sklearn.metrics import confusion_matrix

#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, yhat)

print(cf_matrix)


# In[71]:


import seaborn as sns

text = np.array([['TP', 'FN'], ['FP','TN']])

# combining text with values
formatted_text = (np.array(["{0}\n{1:0}".format(
    text, cf_matrix) for text, cf_matrix in zip(text.flatten(), cf_matrix.flatten())])).reshape(2, 2)
# Reshape data array and text array into 1D using np.flatten().
# Then zip them together to iterate over both text and value.
# Use formatted strings to create customized new value.
# Return a reshaped array of the same size containing customized values.

ax = sns.heatmap(cf_matrix, annot = formatted_text, fmt="", cmap='Reds')

ax.set_title('Confusion matrix');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Churn = 0','Churn = 1'])
ax.yaxis.set_ticklabels(['Churn = 0','Churn = 1'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[80]:


print(classification_report(y_test,yhat))


# In[72]:


precision = 23/(23+2)
precision


# In[74]:


recall = round(23/(23+8),2)
recall


# In[77]:


f1_score = 2*(0.92 * 0.74)/(0.92 + 0.74)
f1_score
# The higher the better 


# Based on the count of each section, we can calculate precision and recall of each label:
# 
# *   **Precision** is a measure of the accuracy provided that a class label has been predicted. It is defined by: precision = TP / (TP + FP)
# 
# *   **Recall** is the true positive rate. It is defined as: Recall =  TP / (TP + FN)
# 
# So, we can calculate the precision and recall of each class.
# 
# **F1 score:**
# Now we are in the position to calculate the F1 scores for each label based on the precision and recall of that label.
# 
# The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. It is a good way to show that a classifer has a good value for both recall and precision.
# 
# Finally, we can tell the average accuracy for this classifier is the average of the F1-score for both labels, which is 0.72 in our case.
# 

# #### log loss: lower the better

# In[79]:


from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)


# In[84]:


# use another method
lr2 = LogisticRegression(C=0.01, solver='sag')
lr2.fit(x_train,y_train)
yhat_prob2 = lr2.predict_proba(x_test)
# logloss: the smaller the better
print ("LogLoss: %.3f " % log_loss(y_test, yhat_prob2))


# #### Solver: ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers
# - newton-cg: best but computionally expensive
# - lbfgs: best for small dataset, as default 
# - liblinear: best for large-scale learning challenge. It applies automatic parameter selection (a.k.a L1 Regularization) and it’s recommended when you have high dimension dataset (recommended for solving large-scale classification problems)
# - sag:It is faster than other solvers for large datasets when both the number of samples and the number of features are large.t only supports L2 penalization
# - saga: The SAGA solver is a variant of SAG that also supports the non-smooth penalty L1 option (i.e. L1 Regularization). This is therefore the solver of choice for sparse multinomial logistic regression. It also has a better theoretical convergence compared to SAG, but more memory cost.

# In[ ]:




