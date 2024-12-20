#!/usr/bin/env python
# coding: utf-8

# In[37]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np; np.random.seed(42)
import seaborn as sns

import pyLDAvis
import pyLDAvis.gensim
from collections import OrderedDict
from itertools import combinations
import MeCab
from konlpy.tag import *
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.preprocessing import normalize;
from sklearn import decomposition;


# In[40]:


# 데이터 읽어오기 - 특정 기간에 한해서

df = pd.read_csv('adoor_data/sns_feed.csv', encoding='UTF8')
df['created_at'] = pd.to_datetime(df['created_at'])

start_date = pd.Timestamp(2019, 1, 28, 0)
end_date = pd.Timestamp(2019, 4, 1, 0)

mask = (df['created_at'] > start_date) & (df['created_at'] <= end_date)
df = df.loc[mask]

df.head()


# In[41]:


# 형태소 분석

twitter = Okt()
pos = lambda d: ['/'.join(p) for p in twitter.pos(d, stem=True, norm=True)]

texts_ko = []
content = df.photo
docs_ko = ""

for row in content:
    text = row
    if not pd.isna(text):
        docs_ko = docs_ko + text
        for morph in pos(text):
            if (morph.split('/')[1]) in ['Noun'] and len(morph.split('/')[0]) > 1:
                m = []
                m.append(morph)
                texts_ko.append(m)


# In[42]:


dictionary_ko = corpora.Dictionary(texts_ko)
dictionary_ko.save('ko.dict')


# In[43]:


# tfidf 변환

tf_ko = [dictionary_ko.doc2bow(text) for text in texts_ko]
tfidf_model_ko = models.TfidfModel(tf_ko)
tfidf_ko = tfidf_model_ko[tf_ko]
corpora.MmCorpus.serialize('ko.mm', tfidf_ko)


# In[44]:


# LDA 모델 - tfidf 변환을 하지 않은 경우

ntopics, nwords = 3, 5

lda_ko = models.ldamodel.LdaModel(tf_ko, id2word=dictionary_ko, num_topics=ntopics)
lda_ko.print_topics()


# In[45]:


# LDA 모델 - tfidf 변환을 적용한 경우

lda_ko = models.ldamodel.LdaModel(tfidf_ko, id2word=dictionary_ko, num_topics=ntopics)
lda_ko.print_topics()


# In[46]:


ldatopics = lda_ko.show_topics(formatted=False)
ldatopics[0]


# In[47]:


# LSI (Latent Semantic Indexing) 모델

lsi_ko = models.lsimodel.LsiModel(tfidf_ko, id2word=dictionary_ko, num_topics=ntopics)
lsi_ko.print_topics()


# In[48]:


# HDP (Hierarchical Dirichlet Process) 모델

hdp_ko = models.hdpmodel.HdpModel(tfidf_ko, id2word=dictionary_ko)
hdp_ko.print_topics(ntopics, nwords)


# In[49]:


# Bag of Words

bow = tfidf_model_ko[dictionary_ko.doc2bow(texts_ko[0])]
sorted(lda_ko[bow], key=lambda x: x[1], reverse=True)


# In[50]:


sorted(hdp_ko[bow], key=lambda x: x[1], reverse=True)


# In[51]:


bow = tfidf_model_ko[dictionary_ko.doc2bow(texts_ko[1])]
sorted(lda_ko[bow], key=lambda x: x[1], reverse=True)


# In[14]:


sorted(hdp_ko[bow], key=lambda x: x[1], reverse=True)


# In[ ]:





# In[15]:


# Coherence 수준 측정 - LDA 모델

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = models.LdaModel(corpus=tf_ko, num_topics=num_topics, id2word=dictionary_ko)
        model_list.append(model)
        coherencemodel = models.CoherenceModel(model=model, texts=texts_ko, dictionary=dictionary_ko, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[16]:


coherence_model_lda = models.CoherenceModel(model=lda_ko, texts=texts_ko, dictionary=dictionary_ko, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[17]:


model_list, coherence_values = compute_coherence_values(
    dictionary=dictionary_ko, 
    corpus=tf_ko, 
    texts=texts_ko, 
    start=1, 
    limit=10, 
    step=1)


# In[18]:


limit=10; start=1; step=1;
x = range(start, limit, step)
plt.rcParams["font.size"] = 20
plt.figure(figsize=(9,5))
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[52]:


for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[20]:


optimal_model = model_list[1]
model_topics = optimal_model.show_topics(formatted=False)
optimal_model.print_topics(num_words=10)


# In[ ]:





# In[35]:


path='/Library/Fonts/NanumGothic.ttf'
font_name = fm.FontProperties(fname=path, size=50).get_name()
matplotlib.rc('font', family=font_name)
matplotlib.rc('axes', unicode_minus=False)
plt.rc('font', family=font_name)
plt.rc('axes', unicode_minus=False)


# In[24]:


# LDA Model w/ 3 Topics

data_lda = {i: OrderedDict(lda_ko.show_topic(i, 5)) for i in range(ntopics)}
df_lda = pd.DataFrame(data_lda)
df_lda = df_lda.fillna(0).T
print(df_lda.shape)
df_lda


# In[36]:


# Cluster Map - LDA 모델

sns.set(font_scale=3)
g=sns.clustermap(df_lda.corr(), center=0, standard_scale=1, cmap="RdBu", metric='cosine', linewidths=.75, figsize=(15, 15))
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()
## 폰트 문제를 해결하지 못했다........ matplotlib와 다른 문제인 것 같다..... 폰트는 더 이상 포기


# In[53]:


# 3-topic LDA 모델 - MDS Mapping

corpus_lda = lda_ko[tfidf_ko]

pyLDAvis.enable_notebook()
panel = pyLDAvis.gensim.prepare(lda_ko, corpus_lda, dictionary_ko, mds='tsne')
panel


# In[ ]:





# In[54]:


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


# In[55]:


vectorizer = CountVectorizer(tokenizer=getNVM_lemma, min_df=2)
x_counts = vectorizer.fit_transform(df['photo'].values.astype('U'))
print( "Created %d X %d document-term matrix" % (x_counts.shape[0], x_counts.shape[1]) )
transformer = TfidfTransformer(smooth_idf=False);
x_tfidf = transformer.fit_transform(x_counts);


# In[56]:


terms = vectorizer.get_feature_names()
print("Vocabulary has %d distinct terms" % len(terms))


# In[57]:


xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
model = NMF(n_components=10, init='nndsvd');
model.fit(xtfidf_norm)


# In[58]:


def get_nmf_topics(model, n_top_words):
    
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {};
    for i in range(num_topics):
        
        words_ids = model.components_[i].argsort()[:-10 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict);


# In[63]:


num_topics = 1
nmf_df = get_nmf_topics(model, 2)
nmf_df


# In[61]:


num_topics = 5
nmf_df = get_nmf_topics(model, 5)
nmf_df


# In[60]:


num_topics = 10
nmf_df = get_nmf_topics(model, 5)
nmf_df


# In[ ]:




