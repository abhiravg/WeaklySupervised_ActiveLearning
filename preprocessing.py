from bs4 import BeautifulSoup
from nltk.tokenize.toktok import ToktokTokenizer
import re
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import check_random_state
from sklearn.preprocessing import scale, MinMaxScaler
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


def pre_processing_imdb(csv_file):
    dataset = pd.read_csv(csv_file)
    for i in range(len(dataset)):
        if dataset["sentiment"][i] == "positive":
            dataset["sentiment"][i] = 1
        else:
            dataset["sentiment"][i] = 0

    dataset["review"] = dataset["review"].apply(preprocess_text)
    dataset["review"] = dataset["review"].apply(remove_special_characters)
    dataset["review"] = dataset["review"].apply(remove_stopwords)
    return dataset
    # labels = dataset["sentiment"].values
    # tfidfvectorizer = TfidfVectorizer()
    # features = tfidfvectorizer.fit_transform(dataset["review"])
    # return features, labels


# def split_data(features, labels):
#     x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=12)
#     return x_train, x_test, y_train, y_test

def split_data(dataframe):
    df_train, df_test = train_test_split(dataframe, test_size=0.2, random_state=12)
    return df_train, df_test


def tfidf_tokenizer(dataframe):
    tfidf = TfidfVectorizer()
    features = tfidf.fit_transform(dataframe["review"])
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