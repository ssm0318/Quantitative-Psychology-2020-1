#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from os import path
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import os
import csv
import textract
import nltk

import MeCab
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from konlpy.tag import *
from collections import Counter
from nltk.corpus import stopwords
import pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('adoor_data/answers.csv', encoding='UTF8')
df['created_at'] = pd.to_datetime(df['created_at'])

start_date = pd.Timestamp(2019, 1, 28, 0)
end_date = pd.Timestamp(2019, 4, 1, 0)

mask = (df['created_at'] > start_date) & (df['created_at'] <= end_date)
df = df.loc[mask]

df.head()


# In[3]:


def getNVM_lemma(text):
    tokenizer = MeCab.Tagger()
    parsed = tokenizer.parse(text)
    word_tag = [w for w in parsed.split("\n")]
    pos = []
    tags = ['NNG','NNP','VV','VA', 'VX', 'VCP','VCN']
    for word_ in word_tag[:-2]:
        word = word_.split("\t")
        tag = word[1].split(",")
        if(len(word[0]) < 2) or ("게" in word[0]):
            continue
        if(tag[-1] != '*'):
            t = tag[-1].split('/')
            if(len(t[0]) > 1 and ('VV' in t[1] or 'VA' in t[1] or 'VX' in t[1])):
                pos.append(t[0])
        else:
            if(tag[0] in tags):
                pos.append(word[0])
    return pos


# In[4]:


tf_vect = TfidfVectorizer(tokenizer=getNVM_lemma,ngram_range=(1, 2), min_df=2, max_df=20000) 
dtm = tf_vect.fit_transform(df['content'].values.astype('U'))

n_topics = 3

lda = LatentDirichletAllocation(n_components=n_topics) 
lda.fit(dtm)


# In[5]:


names = tf_vect.get_feature_names() 
topics = dict() 

for idx, topic in enumerate(lda.components_): 
    vocab = [] 
    for i in topic.argsort()[:-(10-1):-1]: 
        vocab.append((names[i], topic[i].round(2))) 
    print("주제 %d:" % (idx +1)) 
    print([(names[i], topic[i].round(2)) for i in topic.argsort()[:-(10-1):-1]])


# In[6]:


visual = pyLDAvis.sklearn.prepare(lda_model=lda, dtm=dtm, vectorizer=tf_vect) 
pyLDAvis.save_html(visual, 'LDA_Visualization.html') 
pyLDAvis.display(visual)


# In[ ]:




