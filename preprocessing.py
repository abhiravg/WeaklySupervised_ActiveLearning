from bs4 import BeautifulSoup
from nltk.tokenize.toktok import ToktokTokenizer
import re
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import check_random_state
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.datasets import load_files
import numpy as np


# read and process stopwords
def process_stopwords():
    f = open("stopwords")
    wordList = f.readlines()
    stopwords = []

    for stopword in wordList:
        stopwords.append(stopword.strip('\n'))

    tokenizer = ToktokTokenizer()
    return tokenizer, stopwords


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup


def remove_between_square_brackets(text):
    letters_only = re.sub('\[[^]]*\]', " ", str(text))
    return letters_only.lower()


def preprocess_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


def remove_special_characters(text):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


def stemming(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


def remove_stopwords(text):
    tokenizer, stopwords = process_stopwords()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def fetch_and_preprocess(dataset):
    
    if dataset == "IMDB":
        return _pre_processing_imdb()
    
    else:
        raise ValueError(f"The value '{dataset}' for argument `dataset`"
                          " is not recognised.")


def _pre_processing_imdb():
    train = load_files("./imdb/train",
                       categories=["neg", "pos", "unsup"], encoding='utf-8', 
                       random_state=123)
    test = load_files("./imdb/test", categories=["neg", "pos"],
                      encoding='utf-8', random_state=123)
    
    train_df = pd.DataFrame({
        'data': train.data,
        'target': train.target
    })

    test_df = pd.DataFrame({
        'data': test.data,
        'target': test.target
    })

    unlab_df = train_df[train_df.target == 2].reset_index(drop=True)
    train_df = train_df[train_df.target != 2].reset_index(drop=True)

    unlab_df = unlab_df['data']

    # Initial hand labelled sample is 2000. 
    # This is denoted in num_labeled_samples in main.py
    train_gold_df = train_df.groupby('target').sample(n=5000, random_state=123).sample(frac=1).reset_index(drop=True)

    # corpus = train_gold_df['data'].append(unlab_df, ignore_index=True)
    # NOTE (Abhirav) Using strip_html is giving an error in the vectorizer 
    # corpus = corpus.apply(strip_html)

    # X_corpus = tfidf_vectorizer(corpus, max_df=0.4, ngram_range=(1,2))
    
    return train_gold_df, unlab_df, test_df


def fetch_and_preprocess_snorkel(training_file, testing_file):
    train = load_files(training_file,
                       categories=["neg", "pos", "unsup"], encoding='utf-8',
                       random_state=123)
    test = load_files(testing_file, categories=["neg", "pos"],
                      encoding='utf-8', random_state=123)

    train_df = pd.DataFrame({
        'data': train.data,
        'target': train.target
    })

    test_df = pd.DataFrame({
        'data': test.data,
        'target': test.target
    })

    unlab_df = train_df[train_df.target == 2].reset_index(drop=True)
    train_df = train_df[train_df.target != 2].reset_index(drop=True)
    unlab_df = unlab_df['data']
    
    return unlab_df, train_df, test_df


def fetch_and_preprocess_supervised(training_file, testing_file):
    train = load_files(training_file,
                       categories=["neg", "pos", "unsup"], encoding='utf-8',
                       random_state=123)
    test = load_files(testing_file, categories=["neg", "pos"],
                      encoding='utf-8', random_state=123)

    train_df = pd.DataFrame({
        'data': train.data,
        'target': train.target
    })

    test_df = pd.DataFrame({
        'data': test.data,
        'target': test.target
    })
    
    train_df = train_df[train_df['target'] != 2].reset_index(drop=True)
    return train_df, test_df


def split_data(dataframe):
    df_train, df_test = train_test_split(dataframe, test_size=0.2, random_state=12)
    return df_train, df_test


vectorizer = None


def tfidf_vectorizer(data, **kwargs):
    global vectorizer
    if vectorizer == None:
        # print("1")
        vectorizer = TfidfVectorizer(**kwargs)
    features = vectorizer.fit_transform(data)
    return features


def get_k_random_labeled_samples(dataframe, num_labeled_samples):

    # generating random seed
    random_state = check_random_state(0)
    df_train = dataframe.sample(num_labeled_samples, replace=False, random_state=random_state)
    random_samples = df_train.index.values.tolist()
    return random_samples, df_train


def normalize(X_train, X_val, X_test):
    normalizer = MinMaxScaler()
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)
    return X_train, X_val, X_test