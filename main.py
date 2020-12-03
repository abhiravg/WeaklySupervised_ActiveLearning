from preprocessing import fetch_and_preprocess, split_data, get_k_random_labeled_samples, normalize, tfidf_vectorizer
import numpy as np
import pandas as pd
from Training import train, get_test_accuracy
from sample_selection import random_sampling, margin_selection_sampling, entropy_selection_sampling
from weak_supervision import weak_supervisor
from scipy.sparse import vstack

num_labeled_samples = 10000
max_queries = 40000
classification_accuracies = []
initial_labeled_examples = num_labeled_samples

# read and preprocess data
train_gold_df, unlab_df, test_df = fetch_and_preprocess("IMDB")

# initial hand labeled samples
print("train_gold_df: ", train_gold_df.shape)
# rest of the unlabeled corpus
print("unlab_df: ", unlab_df.shape)
# test labeled
print("test_df: ", test_df.shape)

# weak supervision on validation dataset
validation_df = weak_supervisor(pd.DataFrame(unlab_df), "majority_voter")
# print(validation_df.head(20))

# construct word embeddings
corpus = train_gold_df['data'].append(validation_df['data'], ignore_index=True)
X_corpus = tfidf_vectorizer(corpus, max_df=0.4, ngram_range=(1,2), max_features=5000)

X_train = X_corpus[:num_labeled_samples]
Y_train = train_gold_df["target"].values

X_test = tfidf_vectorizer(test_df["data"], max_df=0.4, ngram_range=(1,2), max_features=5000)
Y_test = test_df["target"].values

# construct word embeddings for validation set
X_val = X_corpus[num_labeled_samples:]
# Y_true_val = validation_df["sentiment"].values
Y_weak_val = validation_df["weak_labels"].values

# X_train, X_val, X_test = normalize(X_train, X_val, X_test)
print("X_train shape: ", X_train.shape)
print("X_val shape: ", X_val.shape)
print("X_test shape: ", X_test.shape)

print("Y_train shape: ", Y_train.shape)
print("Y_weak_val shape: ", Y_weak_val.shape)
print("Y_test shape: ", Y_test.shape)

####TESTED TILL HERE####

# TODO Test remaining. Code already in place.

# Building training model
classifier, Y_test_predicted = train(X_train, Y_train, X_test, "NB")
# printing accuracy of the model on the test set
active_iteration = 1
queried_samples = num_labeled_samples
get_test_accuracy(active_iteration, Y_test, Y_test_predicted, classification_accuracies)

# active learning algorithm
while queried_samples < max_queries:
    active_iteration += 1

    # soft labels on validation set
    soft_labels = classifier.predict_proba(X_val)

    # sample low confidence samples
    uncertain_samples = entropy_selection_sampling(soft_labels, num_labeled_samples)

    # sample the uncertain samples from the validation set
    X_new_val = X_val[uncertain_samples, :]
    Y_new_weak_val = Y_weak_val[uncertain_samples]

    print("X_new_val dimensions: ", X_new_val.shape)
    # add the newly sampled validation data into training set
    X_train = vstack([X_train, X_new_val])
    Y_train = np.concatenate([Y_train, Y_new_weak_val], axis=0)

    # remove the newly sampled validation data from existing validation set
    remaining_samples = np.array(list(set(range(X_val.shape[0])) - set(uncertain_samples)))
    X_val = X_val[remaining_samples, :]
    Y_weak_val = Y_weak_val[remaining_samples]

    # incrementing the queries made
    queried_samples += num_labeled_samples
    print("new x_train dimensions: ", X_train.shape)
    print("new y_train dimensions: ", Y_train.shape)
    
    # training model with new training set
    classifier, Y_test_predicted = train(X_train, Y_train, X_test, "RandomForest")
    get_test_accuracy(active_iteration, Y_test, Y_test_predicted, classification_accuracies)

print("classification accuracies: ", classification_accuracies)