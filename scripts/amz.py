#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


# In[2]:


dataset_home = "./amz"


# In[16]:


train = pd.read_csv(f"{dataset_home}/train.csv", names=["target", "title", "data"])
test = pd.read_csv(f"{dataset_home}/test.csv", names=["target", "title", "data"])


# In[17]:


train = train.drop(columns=['title'])
test = test.drop(columns=['title'])


# In[18]:


train.head()


# In[19]:


test.head()


# In[20]:


# NEUTRAL = 0, NEG = 1, POS = 2
train.loc[(train.target == 2),'target'] = 1
train.loc[(train.target == 4),'target'] = 2
train.loc[(train.target == 5),'target'] = 2
train.loc[(train.target == 3),'target'] = 0

test.loc[(test.target == 2),'target'] = 1
test.loc[(test.target == 4),'target'] = 2
test.loc[(test.target == 5),'target'] = 2
test.loc[(test.target == 3),'target'] = 0

train_df = train.groupby('target').sample(n=40000, random_state=123).sample(frac=1)
train_df_index = train_df.index
train_df = train_df.reset_index(drop=True)
test_df = test.groupby('target').sample(n=10000, random_state=123).sample(frac=1).reset_index(drop=True)


# In[22]:


frames = [train_df, test_df]
corpus = pd.concat(frames, ignore_index=True)

vectorizer = TfidfVectorizer(max_df=0.4, ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(corpus['data'])

X_train = tfidf[:train_df.shape[0]]
Y_train = train_df['target'].values

X_test = tfidf[train_df.shape[0]:]
Y_test = test_df['target'].values


# In[25]:


clf = MLPClassifier(random_state=123, max_iter=500, early_stopping=True)
clf.fit(X_train, Y_train)


# In[30]:


clf.score(X_test, Y_test)


# In[31]:


from joblib import dump, load


# In[32]:


dump(clf, 'amz-mlp-1.joblib')


# In[ ]:




