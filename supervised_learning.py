from preprocessing import fetch_and_preprocess_supervised, tfidf_vectorizer
from Training import train, get_test_accuracy
import pandas as pd

classification_accuracies = []
train_df, test_df = fetch_and_preprocess_supervised("./imdb/train", "./imdb/test")
print("train_df shape: ", train_df.shape)
print("test_df shape: ", test_df.shape)

X_train = tfidf_vectorizer(train_df['data'], max_df=0.4, ngram_range=(1,2), max_features=5000)
Y_train = train_df['target'].values
X_test = tfidf_vectorizer(test_df['data'], max_df=0.4, ngram_range=(1,2), max_features=5000)
Y_test = test_df['target'].values

classifier, Y_test_predicted = train(X_train, Y_train, X_test)
get_test_accuracy(1, Y_test, Y_test_predicted, classification_accuracies)