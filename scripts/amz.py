#!/usr/bin/env python
# coding: utf-8


import os
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from joblib import dump, load

from time import localtime, strftime
import logging


def remove_special_characters(text):
  text = text.lower()
  pattern = r'[^a-zA-z0-9\s]'
  text = re.sub(pattern, '', text)
  return text


logging.basicConfig(filename='amz.log', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)


logging.info(f"Application amz started with PID {os.getpid()}")
cur_time = strftime("%Y-%d-%m-%H-%M-%S", localtime())
dataset_home = "/mydata/datasets/amz"
tfidf_path = Path('/mydata/models/amz-tfidf-10-5.joblib')
y_train_path = Path('/mydata/models/amz-y-train.joblib')
y_test_path = Path('/mydata/models/amz-y-test.joblib')


if tfidf_path.exists():
  tfidf = load(tfidf_path)
  y_train = load(train_df['target'].values, y_train_path)
  y_test = load(test_df['target'].values, y_test_path)
else:
  train = pd.read_csv(f"{dataset_home}/train.csv", names=["target", "title", "data"])
  test = pd.read_csv(f"{dataset_home}/test.csv", names=["target", "title", "data"])

  train = train.drop(columns=['title'])
  test = test.drop(columns=['title'])

  # NEUTRAL = 0, NEG = 1, POS = 2
  train.loc[(train.target == 2),'target'] = 1
  train.loc[(train.target == 4),'target'] = 2
  train.loc[(train.target == 5),'target'] = 2
  train.loc[(train.target == 3),'target'] = 0

  test.loc[(test.target == 2),'target'] = 1
  test.loc[(test.target == 4),'target'] = 2
  test.loc[(test.target == 5),'target'] = 2
  test.loc[(test.target == 3),'target'] = 0

  train_df = train.groupby('target').sample(n=60000, random_state=123).sample(frac=1)
  train_df_index = train_df.index
  train_df = train_df.reset_index(drop=True)
  test_df = test.groupby('target').sample(n=10000, random_state=123).sample(frac=1).reset_index(drop=True)

  frames = [train_df, test_df]
  corpus = pd.concat(frames, ignore_index=True)

  vectorizer = TfidfVectorizer(stop_words='english', max_features=100000, ngram_range=(1, 3),
                                preprocessor=remove_special_characters)
  tfidf = vectorizer.fit_transform(corpus['data'])

  dump(tfidf, tfidf_path)
  dump(train_df['target'].values, y_train_path)
  dump(test_df['target'].values, y_test_path)

boundary = 180000
X_train = tfidf[:boundary]
y_train = train_df['target'].values

X_test = tfidf[boundary:]
y_test = test_df['target'].values


clf = MLPClassifier(random_state=123, early_stopping=True, learning_rate_init=0.0001,
                    tol=1e-6)
clf.fit(X_train, y_train)

logging.info(f"n_iter_: {clf.n_iter_}")
logging.info(f"n_layers_: {clf.n_layers_}")
logging.info(f"loss_: {clf.loss_}")
logging.info(f"Mean test accuracy: {clf.score(X_test, y_test)}")

dump(clf, f"/mydata/models/amz-mlp-{cur_time}.joblib")
logging.info(f"Application amz with PID {os.getpid()} completed")
