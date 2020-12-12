from preprocessing import fetch_and_preprocess_data, split_data, tfidf_vectorizer
import numpy as np
import pandas as pd
from Training import train, get_test_accuracy
from sample_selection import sampling_method
from weak_supervision import weak_supervisor
from scipy.sparse import vstack

from sklearn.calibration import CalibratedClassifierCV


def active_learner(initial_labeled_examples, max_queries, learning_model, sample_select,
                    dataset, dataset_home, label_method=None, weak_supervision=False):
    active_learning_benchmarks = {}

    # read and preprocess data
    train_df, test_df, unlab_df = fetch_and_preprocess_data(dataset, dataset_home, unlab=True)
    num_samples = int(initial_labeled_examples/2)
    train_gold_df = train_df.groupby('target').sample(n=num_samples, random_state=123).sample(frac=1).reset_index(drop=True)

    # # initial hand labeled samples
    # print("train_gold_df: ", train_gold_df.shape)
    # # rest of the unlabeled corpus
    # print("unlab_df: ", unlab_df.shape)
    # # test labeled
    # print("test_df: ", test_df.shape)

    if weak_supervision:
        # weak supervision on validation dataset
        validation_df = weak_supervisor(pd.DataFrame(unlab_df), label_method)
        # print(validation_df.head(20))
        Y_val = validation_df["weak_labels"].values
    else:
        validation_df = pd.concat([train_df, train_gold_df]).drop_duplicates(keep=False).reset_index(drop=True)
        validation_df, test_df = split_data(validation_df, test_df, method="active")
        Y_val = validation_df["target"].values


    # construct word embeddings
    corpus = train_gold_df['data'].append([validation_df['data'], test_df['data']], ignore_index=True)
    X_corpus = tfidf_vectorizer(corpus, max_df=0.4, ngram_range=(1, 2))

    X_train = X_corpus[:initial_labeled_examples]
    Y_train = train_gold_df["target"].values

    test_start = train_gold_df.shape[0] + validation_df.shape[0]
    X_test = X_corpus[test_start:]
    Y_test = test_df["target"].values

    # construct word embeddings for validation set
    X_val = X_corpus[initial_labeled_examples:test_start]

    # print("X_train shape: ", X_train.shape)
    # print("X_val shape: ", X_val.shape)
    # print("X_test shape: ", X_test.shape)

    # print("Y_train shape: ", Y_train.shape)
    # print("Y_val shape: ", Y_val.shape)
    # print("Y_test shape: ", Y_test.shape)

    # Building training model
    classifier, Y_test_predicted = train(X_train, Y_train, X_test, learning_model, scenario="active")
    if learning_model in ["SVC", "SGD"] :
        calibrator = CalibratedClassifierCV(classifier, cv="prefit")
        calibrator.fit(X_train, Y_train)
    # printing accuracy of the model on the test set
    active_iteration = 1
    queried_samples = initial_labeled_examples
    active_learning_benchmarks[active_iteration] = get_test_accuracy(active_iteration, Y_test, Y_test_predicted)
    
    # active learning algorithm
    while queried_samples < max_queries:
        active_iteration += 1

        # soft labels on validation set
        if learning_model in ["SVC", "SGD"]:
            soft_labels = calibrator.predict_proba(X_val)
        else:
            soft_labels = classifier.predict_proba(X_val)

        # sample low confidence samples
        uncertain_samples = sampling_method(soft_labels, initial_labeled_examples, sample_select)

        # sample the uncertain samples from the validation set
        X_new_val = X_val[uncertain_samples, :]
        Y_new_val = Y_val[uncertain_samples]

        # print("X_new_val dimensions: ", X_new_val.shape)
        # add the newly sampled validation data into training set
        X_train = vstack([X_train, X_new_val])
        Y_train = np.concatenate([Y_train, Y_new_val], axis=0)

        # remove the newly sampled validation data from existing validation set
        remaining_samples = np.array(list(set(range(X_val.shape[0])) - set(uncertain_samples)))
        X_val = X_val[remaining_samples, :]
        Y_val = Y_val[remaining_samples]

        # incrementing the queries made
        queried_samples += initial_labeled_examples
        # print("New X_train dimensions: ", X_train.shape)
        # print("New Y_train dimensions: ", Y_train.shape)

        # training model with new training set
        # if queried_samples > (0.9 * max_queries):
        #     classifier, Y_test_predicted = train(X_train, Y_train, X_test, learning_model)
        # else:
        classifier, Y_test_predicted = train(X_train, Y_train, X_test, learning_model, scenario="active")

        active_learning_benchmarks[active_iteration] = get_test_accuracy(active_iteration, Y_test, Y_test_predicted)

    return active_learning_benchmarks
