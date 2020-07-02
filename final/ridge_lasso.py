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


labels = pd.read_csv('survey_data/scale_labels.csv')


# In[5]:


labels


# In[6]:


labels['BRCS']


# In[7]:


data = pd.read_csv('survey_data/results_compilation.csv')
data = data.dropna(how='any')


# In[8]:


data.head()


# In[9]:


data.info()


# In[10]:


X = data.iloc[:,26:64].drop(['ADHDRS'], axis=1)
# X = data.iloc[:,26:63].drop(['RFQ', 'RFQ_m', 'RFQ_v'], axis=1)
# X = data.iloc[:,26:63].drop(['BFI_O'], axis=1)
# X = data.iloc[:,26:63].drop(['PCS9'], axis=1)
# X = data.iloc[:,26:63].drop(['FS'], axis=1)
# X = data.iloc[:,26:63].drop(['HEXACO60'], axis=1)
# X = data.iloc[:,26:63].drop(['BFI_O'], axis=1)
X = X.drop(['RFQ', 'RFQ_m', 'RFQ_v', 'BPSSR_I', 'BPSSR_I5', 'BPSSR_E', 'BPSSR_Amb', "ERQ10_e", "ERQ10_c", "SMS"], axis=1)
x = X.to_numpy() # np array


# In[11]:


y = data['ADHDRS']
# y = data['BPSSR']
# y = data['RFQ_m']
# y = data['BFI_O']
# y = data['PCS9']
# y = data['FS']
# y = data['HEXACO60']
# y = data['BFI_O']


# In[12]:


x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)


# In[13]:


alphas = 10**np.linspace(3,-3,100)*0.5


# In[14]:


ridge = Ridge(fit_intercept=True, normalize=True)
coefs = []
errors = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(x_train, y_train)
    pred = ridge.predict(x_test)
    coefs.append(pd.Series(ridge.coef_, index = X.columns))
    errors.append(mean_squared_error(y_test, pred))


# In[15]:


ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('lambda')
plt.ylabel('coefficients')
plt.title('Ridge')


# In[16]:


ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('lambda')
plt.ylabel('mean squared error')
plt.title('Ridge')


# In[17]:


ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', fit_intercept=True, normalize=True)
ridgecv.fit(x_train, y_train)
ridgecv.alpha_

ridge_fit = Ridge(alpha = ridgecv.alpha_, normalize = True)
ridge_fit.fit(x_train, y_train)
mean_squared_error(y_test, ridge_fit.predict(x_test))

ridge_fit.fit(x, y)
pd.Series(ridge_fit.coef_, index = X.columns)


# In[18]:


ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', fit_intercept=True, store_cv_values = True, normalize=True)
ridgecv.fit(x_train, y_train)
ridgecv.alpha_


# In[19]:


ridge_fit = Ridge(alpha = ridgecv.alpha_, normalize=True)
ridge_fit.fit(x_train, y_train)
mean_squared_error(y_test, ridge_fit.predict(x_test))


# In[20]:


ridge_fit.fit(x, y)
pd.Series(ridge_fit.coef_, index = X.columns)


# In[21]:


print(labels['BFI_C'][0])
print(labels['BFI_E'][0])
print(labels['NPI16'][0])
print(labels['CAMSR'][0])
print(labels['ZBS'][0])


# In[22]:


ax = plt.gca()
ax.plot(alphas, ridgecv.cv_values_.mean(axis=0))
ax.set_xscale('log')
ax.axvline(x=ridgecv.alpha_, color='r')
plt.axis('tight')
plt.xlabel('lambda')
plt.ylabel('mean squared error')
plt.title('Ridge CV')


# In[23]:


lasso = Lasso(max_iter = 10000, normalize = True)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(x_train), y_train)
    coefs.append(lasso.coef_)


# In[24]:


ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('lambda')
plt.ylabel('coefficients')
plt.title('Lasso')


# In[28]:


lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(x_train, y_train)
lassocv.alpha_


# In[29]:


lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(x_train, y_train)
mean_squared_error(y_test, lasso.predict(x_test))


# In[26]:


pd.Series(lasso.coef_, index=X.columns)


# In[27]:


print(labels['BFI_C'][0])
print(labels['BPSSR'][0])
print(labels['CAMSR'][0])
print(labels['ZBS'][0])


# In[ ]:




