#!/usr/bin/env python
# coding: utf-8

# In[1]:


# reference1: https://harvard-iacs.github.io/2019-CS109B/labs/lab7/solutions/
# reference2: https://notebooks.azure.com/api/user/ayoadegoke/project/ISLR-Python/html/Chapter%2010.ipynb#x_10.5.3-Hierarchical-Clustering


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-white')


# In[3]:


df = pd.read_csv('USArrests.csv', index_col=0)
df['Assault'] = df['Assault'].astype('float64')
df['UrbanPop'] = df['UrbanPop'].astype('float64')

df.head()


# In[4]:


X = pd.DataFrame(scale(df), index=df.index, columns=df.columns)


# In[5]:


plt.figure(figsize=(12, 8.5))
dist_mat = pdist(X, metric="correlation")
ward_data = hierarchy.ward(dist_mat)

hierarchy.dendrogram(ward_data);


# In[6]:


scaled_df.inde


# In[ ]:





# In[7]:


# PCA

pca_loadings = pd.DataFrame(PCA().fit(X).components_.T, index=df.columns, columns=['V1', 'V2', 'V3', 'V4'])
pca_loadings


# In[8]:


# Fit the PCA model and transform X to get the principal components
pca = PCA()
df_plot = pd.DataFrame(pca.fit_transform(X), columns=['PC1', 'PC2', 'PC3', 'PC4'], index=X.index)
df_plot


# In[9]:


fig , ax1 = plt.subplots(figsize=(9,7))

ax1.set_xlim(-3.5,3.5)
ax1.set_ylim(-3.5,3.5)

# Plot Principal Components 1 and 2
for i in df_plot.index:
    ax1.annotate(i, (df_plot.PC1.loc[i], -df_plot.PC2.loc[i]), ha='center')

# Plot reference lines
ax1.hlines(0,-3.5,3.5, linestyles='dotted', colors='grey')
ax1.vlines(0,-3.5,3.5, linestyles='dotted', colors='grey')

ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')
    
# Plot Principal Component loading vectors, using a second y-axis.
ax2 = ax1.twinx().twiny() 

ax2.set_ylim(-1,1)
ax2.set_xlim(-1,1)
ax2.tick_params(axis='y', colors='orange')
ax2.set_xlabel('Principal Component loading vectors', color='orange')

# Plot labels for vectors. Variable 'a' is a small offset parameter to separate arrow tip and text.
a = 1.07  
for i in pca_loadings[['V1', 'V2']].index:
    ax2.annotate(i, (pca_loadings.V1.loc[i]*a, -pca_loadings.V2.loc[i]*a), color='orange')

# Plot vectors
ax2.arrow(0,0,pca_loadings.V1[0], -pca_loadings.V2[0])
ax2.arrow(0,0,pca_loadings.V1[1], -pca_loadings.V2[1])
ax2.arrow(0,0,pca_loadings.V1[2], -pca_loadings.V2[2])
ax2.arrow(0,0,pca_loadings.V1[3], -pca_loadings.V2[3]);


# In[10]:


# Standard deviation of the four principal components
np.sqrt(pca.explained_variance_)


# In[11]:


pca.explained_variance_


# In[12]:


pca.explained_variance_ratio_


# In[13]:


plt.figure(figsize=(7,5))

plt.plot([1,2,3,4], pca.explained_variance_ratio_, '-o', label='Individual component')
plt.plot([1,2,3,4], np.cumsum(pca.explained_variance_ratio_), '-s', label='Cumulative')

plt.ylabel('Proportion of Variance Explained')
plt.xlabel('Principal Component')
plt.xlim(0.75,4.25)
plt.ylim(0,1.05)
plt.xticks([1,2,3,4])
plt.legend(loc=2);


# In[ ]:





# In[14]:


# K-means Clustering

np.random.seed(4)
km2 = KMeans(n_clusters=3, n_init=20)
km2.fit(X)


# In[15]:


pd.Series(km2.labels_).value_counts()


# In[16]:


km2.cluster_centers_


# In[17]:


km2.labels_


# In[18]:


# Sum of distances of samples to their closest cluster center.
km2.inertia_


# In[21]:


ax = plt.gca()
ax.scatter(X[:,0], X[:,1], s=40, c=km2.labels_, cmap=plt.cm.prism) 
ax.set_title('K-Means Clustering Results with K=3')
ax.scatter(km2.cluster_centers_[:,0], km2.cluster_centers_[:,1], marker='+', s=100, c='k', linewidth=2);


# In[ ]:




