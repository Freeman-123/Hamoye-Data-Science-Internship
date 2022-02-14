#!/usr/bin/env python
# coding: utf-8

# # Hamoye Winter Data Science Internship
# # Stage B Quiz
# Loading Python Libraries

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Loading the Energy Dataset

# In[2]:


df = pd.read_csv("energydata_complete.csv", encoding="ISO-8859–1")


# In[4]:


df


# Question 12

# In[23]:


df['T2'].shape
x = df.T2.values


# In[24]:


df['T6'].shape
y = df.T6.values


# In[25]:


length = 19735
x = x.reshape(length,1)
y = y.reshape(length,1)


# In[26]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# In[27]:


from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()

linear_model.fit(x_train, y_train)


# In[28]:


predicted_values = linear_model.predict(x_test)


# R - Squared

# In[29]:


from sklearn.metrics import r2_score
r2_score = r2_score(y_test, predicted_values)
round(r2_score, 2)


# Question 13

# In[31]:


df = df.drop(columns=['date', 'lights'])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normalised_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
features_df = normalised_df.drop(columns=['Appliances'])
target = normalised_df['Appliances']


# In[32]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features_df, target, test_size=0.3, random_state=42)


# In[33]:


from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()


linear_model.fit(x_train, y_train)


# In[34]:


predicted_values = linear_model.predict(x_test)


# Mean Absolute Error

# In[35]:


from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predicted_values)
round(mae, 2)


# Question 14; Residual Sum of Squares (RSS)

# In[36]:


rss = np.sum(np.square(y_test - predicted_values))
round(rss, 2) 


# Question15; Root Mean Squared Error (RMSE)

# In[37]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rmse, 3) 


# Question 16; Coefficient of Determination (R-squared)

# In[38]:


from sklearn.metrics import r2_score
r2_score = r2_score(y_test, predicted_values)
round(r2_score, 2) 


# Question 17; Linear weights of model

# In[39]:


def get_weights_df(model, feat, col_name):
    weights = pd.Series(model.coef_, feat.columns).sort_values()
    weights_df = pd.DataFrame(weights).reset_index()
    weights_df.columns = ['Features', col_name]
    weights_df[col_name].round(3)
    return weights_df


# In[40]:


linear_model_weights = get_weights_df(linear_model, x_train, 'Linear_Model_Weight')
linear_model_weights


# Question 18; Ridge Regression

# In[41]:


from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=0.4)
ridge_reg.fit(x_train, y_train)


# In[43]:


predicted_values_ridge = ridge_reg.predict(x_test)


# Root Mean Square Error (Ridge Model)

# In[44]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predicted_values_ridge))
round(rmse, 3)


# Question 19; Lasso Regression

# In[45]:


from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(x_train, y_train)


# In[46]:


lasso_weights_df = get_weights_df(lasso_reg, x_train, 'Lasso_weight')
lasso_weights_df


# Question 20; Root Mean Square Error (Lasso Model)

# In[47]:


predicted_values_lasso = lasso_reg.predict(x_test)

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predicted_values_lasso))
round(rmse, 3)


# In[ ]:




