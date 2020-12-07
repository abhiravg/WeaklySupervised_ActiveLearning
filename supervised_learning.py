from preprocessing import fetch_and_preprocess_supervised, tfidf_vectorizer
from Training import supervised_learning_train, get_test_accuracy
import pandas as pd


def supervised_learn(training_file, testing_file, learning_model):
    # training_file = "./imdb/train" , testing_file = "./imdb/test"
    train_df, test_df = fetch_and_preprocess_supervised(training_file, testing_file)
    print("train_df shape: ", train_df.shape)
    print("test_df shape: ", test_df.shape)

    X_train = tfidf_vectorizer(train_df['data'], max_df=0.4, ngram_range=(1, 2), max_features=5000)
    Y_train = train_df['target'].values
    X_test = tfidf_vectorizer(test_df['data'], max_df=0.4, ngram_range=(1, 2), max_features=5000)
    Y_test = test_df['target'].values

    classifier, Y_test_predicted = supervised_learning_train(X_train, Y_train, X_test, learning_model)
    supervised_learning_benchmarks = get_test_accuracy(1, Y_test, Y_test_predicted)
    return supervised_learning_benchmarks
