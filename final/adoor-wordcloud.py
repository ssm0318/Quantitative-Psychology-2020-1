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

df = pd.read_csv('adoor_data/answers.csv', encoding='UTF8')
df['created_at'] = pd.to_datetime(df['created_at'])
# df = df[(df.author_id != 5)]


# In[4]:


start_date = pd.Timestamp(2019, 1, 28, 0)
end_date = pd.Timestamp(2019, 4, 1, 0)

mask = (df['created_at'] > start_date) & (df['created_at'] <= end_date)
df = df.loc[mask]

df.head()


# In[5]:


content = df.content

for row in content:
    text = row
    if not pd.isna(text):
        morphs.append(twitter.pos(text))


# In[6]:


print(morphs)


# In[7]:


noun_adj_adv_list=[]
 
for sentence in morphs :
    for word, tag in sentence :
        if tag in ['Noun'] and len(word) > 1:
            noun_adj_adv_list.append(word)


# In[8]:


print(noun_adj_adv_list)


# In[9]:


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




