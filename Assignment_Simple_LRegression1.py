#!/usr/bin/env python
# coding: utf-8

# # Q1

# In[1]:


#Delivery_time -> Predict delivery time using sorting time.
#Build a simple linear regression model by performing EDA and do necessary transformations and
#select the best model using R or Python.
#EDA and Data Visualization, Feature Engineering, Correlation Analysis, Model Building, 
#Model Testing and Model Predictions using simple linear regression.


# In[20]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[5]:


# import dataset
dataset=pd.read_csv(r'C:\Users\anupa\Downloads\delivery_time.csv')
dataset


# In[6]:


#EDA and Data Visualization
dataset.info()


# In[7]:


sns.distplot(dataset['Delivery Time'])


# In[8]:


sns.distplot(dataset['Sorting Time'])


# In[9]:


#Feature Engineering
# Renaming Columns
dataset=dataset.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
dataset


# In[10]:


#Correlation Analysis
dataset.corr()


# In[11]:


sns.regplot(x=dataset['sorting_time'],y=dataset['delivery_time'])


# In[12]:


#Model Building
model=smf.ols("delivery_time~sorting_time",data=dataset).fit()


# In[13]:


#Model Testing
# Finding Coefficient parameters
model.params


# In[14]:


# Finding tvalues and pvalues
model.tvalues , model.pvalues


# In[15]:


# Finding Rsquared Values
model.rsquared , model.rsquared_adj


# In[16]:


#Model Predictions
# Manual prediction for say sorting time 5
delivery_time = (6.582734) + (1.649020)*(5)
delivery_time


# In[17]:


# Automatic Prediction for say sorting time 5, 8
new_data=pd.Series([5,8])
new_data


# In[22]:


data_pred=pd.DataFrame(new_data,columns=['sorting_time'])
data_pred


# In[23]:


model.predict(data_pred)


# In[ ]:




