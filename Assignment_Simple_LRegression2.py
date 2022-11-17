#!/usr/bin/env python
# coding: utf-8

# # Q2

# In[1]:


#Salary_hike -> Build a prediction model for Salary_hike 
#Build a simple linear regression model by performing EDA and  do necessary transformations 
#and select the best model using R or Python. EDA and Data Visualization.
#Correlation Analysis. Model Building. Model Testing. Model Predictions.


# In[2]:


# impoort libraries
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[3]:


# import dataset
dataset=pd.read_csv(r'C:\Users\anupa\Downloads\Salary_Data.csv')
dataset


# In[5]:


#EDA and Data Visualization
dataset.info()


# In[7]:


sns.distplot(dataset['YearsExperience'])


# In[8]:


sns.distplot(dataset['Salary'])


# In[9]:


#Correlation Analysis
dataset.corr()


# In[10]:


sns.regplot(x=dataset['YearsExperience'],y=dataset['Salary'])


# In[11]:


#Model Building
model=smf.ols("Salary~YearsExperience",data=dataset).fit()


# In[12]:


#Model Testing
# Finding Cefficient Parameters
model.params


# In[13]:


# Finding Pvalues and tvalues
model.tvalues, model.pvalues


# In[14]:


# Finding Rsquared values
model.rsquared , model.rsquared_adj


# In[15]:


#Model Predictions
# Manual prediction for say 3 Years Experience
Salary = (25792.200199) + (9449.962321)*(3)
Salary


# In[16]:


# Automatic Prediction for say 3 & 5 Years Experience 
new_data=pd.Series([3,5])
new_data


# In[17]:


data_pred=pd.DataFrame(new_data,columns=['YearsExperience'])
data_pred


# In[19]:


model.predict(data_pred)

