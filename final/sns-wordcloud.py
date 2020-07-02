#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import os
import textract
from konlpy.tag import *
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import csv
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from IPython.display import set_matplotlib_formats
matplotlib.rc('font',family = 'Malgun Gothic')
set_matplotlib_formats('retina')
matplotlib.rc('axes',unicode_minus = False)


# In[3]:


twitter = Okt()

text = ""
morphs = []

df = pd.read_csv('adoor_data/sns_feed.csv', encoding='UTF8')

# df = df[(df.adoor != 5)]

df.head()


# In[4]:


content = df.content
tags = df.photo

for row in content:
    text = row
    if not pd.isna(text):
        morphs.append(twitter.pos(text))
        
for row in tags:
    text = row
    if not pd.isna(text):
        morphs.append(twitter.pos(text))


# In[5]:


print(morphs)


# In[6]:


noun_adj_adv_list=[]
 
for sentence in morphs :
    for word, tag in sentence :
        if tag in ['Noun'] and len(word) > 1:
#         if tag in ['Noun'] and ("것" not in word) and ("내" not in word)and ("나" not in word)and ("그" not in word) and ("수"not in word) and("게"not in word)and("말"not in word)and("거" not in word) and ("생각" not in word) and ("사람" not in word):
            noun_adj_adv_list.append(word)


# In[7]:


print(noun_adj_adv_list)


# In[8]:


count = Counter(noun_adj_adv_list)
words = dict(count.most_common())
words


# In[10]:


wordcloud = WordCloud(font_path = '~/Library/Fonts/malgun.ttf', background_color='white',colormap = "Accent_r",
                      width=1500, height=1000).generate_from_frequencies(words)

plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:




