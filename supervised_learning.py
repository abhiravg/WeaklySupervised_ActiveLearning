from preprocessing import fetch_and_preprocess_supervised, tfidf_vectorizer, split_data
from Training import supervised_learning_train, get_test_accuracy
import pandas as pd


def supervised_learn(training_file, testing_file, learning_model):
    # training_file = "./imdb/train" , testing_file = "./imdb/test"
    train_df, test_df = fetch_and_preprocess_supervised(training_file, testing_file)
    frames = [train_df, test_df]
    corpus = pd.concat(frames, ignore_index=True)
    train_df, test_df = split_data(corpus)

    print("train_df shape: ", train_df.shape)
    print("test_df shape: ", test_df.shape)

    frames = [train_df, test_df]
    corpus = pd.concat(frames, ignore_index=True)
    tfidf = tfidf_vectorizer(corpus['data'], max_df=0.4, ngram_range=(1, 2))

    X_train = tfidf[:train_df.shape[0]]
    Y_train = train_df['target'].values
    X_test = tfidf[train_df.shape[0]:]
    Y_test = test_df['target'].values

    classifier, Y_test_predicted = supervised_learning_train(X_train, Y_train, X_test, learning_model)
    supervised_learning_benchmarks = get_test_accuracy(1, Y_test, Y_test_predicted)
    return supervised_learning_benchmarks
