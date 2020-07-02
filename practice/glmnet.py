#!/usr/bin/env python
# coding: utf-8

# In[1]:


# reference: http://www.science.smith.edu/~jcrouser/SDS293/labs/lab10-py.html


# In[2]:


from matplotlib.pylab import rcParams
from matplotlib.ticker import FormatStrFormatter
rcParams['figure.figsize'] = 12, 7.5
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import e

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# In[4]:


data = pd.read_csv('OnlineNewsPopularity.csv')


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


x = data.iloc[:,2:60].to_numpy() # np array
X = data.iloc[:,2:60] # df


# In[8]:


y = data.iloc[:,60]


# In[9]:


x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)


# In[10]:


alphas = 10**np.linspace(6,-2,1000)*0.5


# In[11]:


ridge = Ridge(fit_intercept=True, normalize=True)
coefs = []
errors = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(x_train, y_train)
    pred = ridge.predict(x_test)
    coefs.append(pd.Series(ridge.coef_, index = X.columns))
    errors.append(mean_squared_error(y_test, pred))


# In[12]:


ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log', basex=e)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.axis('tight')
plt.xlabel('log lambda')
plt.ylabel('coefficients')
plt.title('Ridge')


# In[13]:


ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log', basex=e)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.axis('tight')
plt.xlabel('log lambda')
plt.ylabel('mean squared error')
plt.title('Ridge')


# In[14]:


ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', fit_intercept=True, normalize=True)
ridgecv.fit(x_train, y_train)
ridgecv.alpha_

ridge_fit = Ridge(alpha = ridgecv.alpha_, normalize = True)
ridge_fit.fit(x_train, y_train)
mean_squared_error(y_test, ridge_fit.predict(x_test))

ridge_fit.fit(x, y)
pd.Series(ridge_fit.coef_, index = X.columns)


# In[15]:


ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', fit_intercept=True, store_cv_values = True, normalize=True)
ridgecv.fit(x_train, y_train)
ridgecv.alpha_


# In[16]:


ridge_fit = Ridge(alpha = ridgecv.alpha_, normalize=True)
ridge_fit.fit(x_train, y_train)
mean_squared_error(y_test, ridge_fit.predict(x_test))


# In[17]:


ridge_fit.fit(x, y)
pd.Series(ridge_fit.coef_, index = X.columns)


# In[18]:


ax = plt.gca()
ax.plot(alphas, ridgecv.cv_values_.mean(axis=0))
ax.set_xscale('log', basex=e)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.axvline(x=ridgecv.alpha_, color='r')
plt.axis('tight')
plt.xlabel('log lambda')
plt.ylabel('mean squared error')
plt.title('Ridge CV')


# In[ ]:





# In[19]:


lasso = Lasso(max_iter = 10000, normalize = True)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(x_train), y_train)
    coefs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


# In[22]:


ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log', basex=e)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.axis('tight')
plt.xlabel('log lambda')
plt.ylabel('coefficients')
plt.title('Lasso')


# In[20]:


lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(x_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(x_train, y_train)
mean_squared_error(y_test, lasso.predict(x_test))


# In[21]:


pd.Series(lasso.coef_, index=X.columns)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




