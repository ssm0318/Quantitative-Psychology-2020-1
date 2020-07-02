#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Start with loading all necessary libraries
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
import joblib

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

# df = df[df['author_id'] != 5]

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


# In[8]:


tf_vect = CountVectorizer(tokenizer=getNVM_lemma, min_df=2, max_df=6000, max_features=25000)
dtm = tf_vect.fit_transform(df['content'].values.astype('U'))

n_topics = 5

lda = LatentDirichletAllocation(n_components=n_topics, topic_word_prior=0.01, doc_topic_prior=0.001)
lda.fit(dtm)
saved_model = joblib.dump(dtm, 'LDA_IP.pkl')


# In[9]:


names = tf_vect.get_feature_names()
topics_word = dict()
n_words = 10

for idx, topic in enumerate(lda.components_):
    vocab = []
    for i in topic.argsort()[:-(n_words-1):-1]:
        vocab.append((names[i], topic[i].round(2)))
    topics_word[idx+1] = [(names[i], topic[i].round(2)) for i in topic.argsort()[:-(n_words-1):-1]]
max_dict = dict()
for idx, vec in enumerate(lda.transform(dtm)):
    t = vec.argmax()
    if(t not in max_dict):
        max_dict[t] = (vec[t], idx)
    else:
        if(max_dict[t][0] < vec[t]):
            max_dict[t] = (vec[t], idx)
            
sorted_review = sorted(max_dict.items(), key = lambda x: x[0], reverse=False)

for key, value in sorted_review:
    print('주제 {}: {}'.format(key+1, topics_word[key+1]))
    print('[주제 {}의 대표 리뷰 :{}]\n{}\n\n'.format(key+1, value[0], df['content'].values.astype('U')[value[1]]))


# In[10]:


visual = pyLDAvis.sklearn.prepare(lda_model=lda, dtm=dtm, vectorizer=tf_vect)
pyLDAvis.save_html(visual, 'LDA_Visualization2.html')
pyLDAvis.display(visual)

