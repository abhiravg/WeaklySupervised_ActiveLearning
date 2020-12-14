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
    return soup.get_text()


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


def fetch_and_preprocess_data(dataset, dataset_home, unlab=False):
    # unlab is whether to return unlabelled data
    if dataset == "IMDB":
        return _pre_processing_imdb(dataset_home, unlab)
    elif dataset == "AMZ":
        return _pre_processing_amz(dataset_home)
    elif dataset == "YELP":
        return _pre_processing_yelp(dataset_home)
    else:
        raise ValueError(f"The value '{dataset}' for argument `dataset`"
                         " is not recognised.")


def _pre_processing_amz(dataset_home):
    train = pd.read_csv(f"{dataset_home}/train.csv", names=["target", "title", "data"])
    test = pd.read_csv(f"{dataset_home}/test.csv", names=["target", "title", "data"])

    train = train.drop(columns=['title'])
    test = test.drop(columns=['title'])

    train_df = train.groupby('target').sample(n=20000, random_state=123).sample(frac=1).reset_index(drop=True)
    test_df = test.groupby('target').sample(n=5000, random_state=123).sample(frac=1).reset_index(drop=True)

    return train_df, test_df


def _pre_processing_yelp(dataset_home, unlab):
    train = pd.read_csv(f"{dataset_home}/train.csv", names=["target", "data"])
    test = pd.read_csv(f"{dataset_home}/test.csv", names=["target", "data"])

    train_df = train.groupby('target').sample(n=50000, random_state=123).sample(frac=1)
    train_df_index = train_df.index
    train_df = train_df.reset_index(drop=True)
    test_df = test.groupby('target').sample(n=10000, random_state=123).sample(frac=1).reset_index(drop=True)

    if unlab:
        train = train.drop(index=train_df_index)
        unlab_df = train.groupby('target').sample(n=50000, random_state=123).sample(frac=1).reset_index(drop=True)
        return train_df, test_df, unlab_df
    else:
        return train_df, test_df


def _pre_processing_imdb(dataset_home, unlab):
    train = load_files(f"{dataset_home}/train",
                       categories=["neg", "pos", "unsup"], encoding='utf-8',
                       random_state=123)
    test = load_files(f"{dataset_home}/test", categories=["neg", "pos"],
                      encoding='utf-8', random_state=123)

    train_df = pd.DataFrame({
        'data': train.data,
        'target': train.target
    })

    test_df = pd.DataFrame({
        'data': test.data,
        'target': test.target
    })

    train_df['data'] = train_df['data'].apply(strip_html)
    test_df['data'] = test_df['data'].apply(strip_html)

    unlab_df = train_df[train_df.target == 2].reset_index(drop=True)
    train_df = train_df[train_df.target != 2].reset_index(drop=True)

    unlab_df = pd.DataFrame(unlab_df['data'])

    if unlab:
        return train_df, test_df, unlab_df
    else:
        return train_df, test_df


def _split_data(dataframe):
    df_train, df_test = train_test_split(dataframe, test_size=0.2, random_state=123)
    return df_train, df_test


def split_data(train_df, test_df, unlab_df=None, method="supervised"):
    if method == "supervised":
        frames = [train_df, test_df]
        corpus = pd.concat(frames, ignore_index=True)
        train_df, test_df = _split_data(corpus)
        frames = [train_df, test_df]
        boundary = train_df.shape[0]
        Y_train = train_df['target'].values
    elif method == "weak_supervision":
        frames = [unlab_df, test_df]
        boundary = unlab_df.shape[0]
        Y_train = unlab_df['target'].values
    else:
        frames = [train_df, test_df]
        corpus = pd.concat(frames, ignore_index=True)
        train_df, test_df = _split_data(corpus)
        return train_df, test_df

    # print("train_df shape: ", train_df.shape)
    # print("test_df shape: ", test_df.shape)

    corpus = pd.concat(frames, ignore_index=True)
    tfidf = tfidf_vectorizer(corpus['data'], max_df=0.4, ngram_range=(1, 2))

    X_train = tfidf[:boundary]
    
    X_test = tfidf[boundary:]
    Y_test = test_df['target'].values

    return X_train, Y_train, X_test, Y_test


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
    df_train = dataframe.sample(
        num_labeled_samples, replace=False, random_state=random_state)
    random_samples = df_train.index.values.tolist()
    return random_samples, df_train


def normalize(X_train, X_val, X_test):
    normalizer = MinMaxScaler()
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)
    return X_train, X_val, X_test
