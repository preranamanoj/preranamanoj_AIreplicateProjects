#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv("Salary.txt")


# In[7]:


#dataset


# In[21]:


x=dataset.iloc[:,:1].values
#x


# In[22]:


y=dataset.iloc[:,1:].values
#y


# In[23]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(x,y, color='r')


# In[24]:


from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[25]:


from sklearn.linear_model import LinearRegression


# In[26]:


regressor=LinearRegression()


# In[27]:


regressor.fit(x_train,y_train)


# In[28]:


y_pred=regressor.predict(x_test)


# In[29]:


y_pred


# In[30]:


y_test


# In[31]:


plt.scatter(x,y,color='r')
plt.plot(x, regressor.predict(x),color='blue')


# In[32]:


from sklearn.preprocessing import PolynomialFeatures


# In[33]:


poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x)


# In[34]:


regressor.fit(x_poly,y)


# In[35]:


plt.scatter(x,y,color='r')
plt.plot(x, regressor.predict(poly.fit_transform(x)),color='blue')


# In[36]:


y_pred=regressor.predict(poly.fit_transform(x))


# In[37]:


y_pred


# In[63]:


#y

