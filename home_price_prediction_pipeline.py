#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


# In[2]:


df = pd.read_csv(r"C:\Gowtham\Finger tips\All Projects\Python + ML\ML Linear Regression Home Price Prediction & Analysis Project\Melbourne_housing_FULL.csv")


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df = df.drop(['Address','Date','Postcode','YearBuilt',"Lattitude","Longtitude"],axis=1)


# In[8]:


df


# In[9]:


df.shape


# In[10]:


df.isnull().sum()


# In[11]:


df[['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']] = df[['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']].fillna(0)
df['Landsize'] = df['Landsize'].fillna(df.Landsize.mean())
df['BuildingArea'] = df['BuildingArea'].fillna(df.BuildingArea.mean())


# In[12]:


df.dropna(inplace=True)


# In[13]:


df.shape


# In[14]:


df.info()


# In[15]:


df['Method'].unique()


# In[16]:


df['SellerG'].unique()


# In[17]:


df['CouncilArea'].unique()


# In[18]:


df['Regionname'].unique()


# In[19]:


df = pd.get_dummies(df, drop_first=True)


# In[20]:


df.shape


# In[21]:


df.head()


# In[22]:


X = df.drop('Price', axis=1)
y = df['Price']


# In[23]:


X.head()


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[25]:


X_train.shape


# In[26]:


X_test.shape


# In[27]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)


# In[28]:


reg.score(X_test, y_test)


# In[29]:


reg.score(X_train, y_train)


# In[ ]:





# # Ridge Regression
# 
# alpha = Regularization strength; must be a positive float. Regularization
#     improves the conditioning of the problem and reduces the variance of
#     the estimates. Larger values specify stronger regularization.
# max_iter : Maximum number of iterations for conjugate gradient solver.

# In[30]:


ridge_reg= Ridge(alpha=50, max_iter=100,)
ridge_reg.fit(X_train, y_train)


# In[31]:


ridge_reg.score(X_test, y_test)


# In[32]:


ridge_reg.score(X_train, y_train)


# # Lasso Regression

# In[33]:


lasso_reg = Lasso(alpha=50, max_iter=100)
lasso_reg.fit(X_train, y_train)


# In[34]:


lasso_reg.score(X_test, y_test)


# In[35]:


lasso_reg.score(X_train, y_train)


# In[ ]:





# In[ ]:




