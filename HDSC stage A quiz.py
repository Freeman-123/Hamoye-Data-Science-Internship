#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("FoodBalanceSheets_E_Africa_NOFLAG.csv", encoding="ISO-8859–1")


# In[4]:


df


# In[7]:


#Question 11
df.groupby('Item')['Y2014'].sum()


# In[12]:


#Question 12
df.groupby('Item')['Y2017'].sum()


# In[13]:


#Question 12
df.describe(include = 'all')


# In[14]:


#Question 13
df.isnull().sum()


# In[15]:


#Question 13
df.isnull().sum()/len(df) * 100


# In[18]:


#Question 15
df.groupby('Element').get_group('Import Quantity').sum()


# In[19]:


#Question 16
df.groupby('Element').get_group('Production').sum()


# In[28]:


#Question 17 & 18
df.groupby('Element')['Y2018'].sum()


# In[41]:


#Question 19
df.groupby(['Area','Element']).get_group(('Algeria', 'Import Quantity')).sum()


# In[43]:


#Question 20
len(df.groupby('Area').count())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




