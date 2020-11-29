from preprocessing import pre_processing_imdb, split_data, get_k_random_labeled_samples, normalize, tfidf_tokenizer
import numpy as np
from Training import train, get_test_accuracy
from sample_selection import random_sampling, margin_selection_sampling, entropy_selection_sampling
from weak_supervision import weak_supervisor

num_labeled_samples = 500
max_queries = 2000
classification_accuracies = []
initial_labeled_examples = num_labeled_samples

# read and preprocess data
dataframe = pre_processing_imdb("IMDB_Dataset.csv")
df_training, df_test = split_data(dataframe)
print("df_train shape: ", df_training.shape)

# get random labeled samples
random_samples, df_train = get_k_random_labeled_samples(df_training, num_labeled_samples)
print("df_train: ", df_train.shape)
df_validation = df_training.drop(random_samples)
print("df_validation: ", df_validation.shape)

queried_samples = num_labeled_samples
# construct word embeddings
X_train = tfidf_tokenizer(df_train)
Y_train = df_train["sentiment"].values

X_test = tfidf_tokenizer(df_test)
Y_test = df_test["sentiment"].values

# weak supervision on validation dataset
df_validation = weak_supervisor(df_validation, "label_model")
print(df_validation.head(20))

# construct word embeddings for validation set
X_val = tfidf_tokenizer(df_validation)
Y_true_val = df_validation["sentiment"].values
Y_weak_val = df_validation["weak_labels"].values

# X_train, X_val, X_test = normalize(X_train, X_val, X_test)
print("X_train shape: ", X_train.shape)
print("X_val shape: ", X_val.shape)
print("X_test shape: ", X_test.shape)

# Building training model
classifier, X_train, Y_train, Y_test_predicted, Y_val_predicted = train(X_train, Y_train, X_val, X_test)
# printing accuracy of the model on the test set
get_test_accuracy(1, Y_test, Y_test_predicted, classification_accuracies)
active_iteration = 1

# active learning algorithm
while queried_samples < max_queries:
    active_iteration += 1

    # soft labels on validation set
    soft_labels = classifier.predict_proba(X_val)

    # sample low confidence samples
    uncertain_samples = margin_selection_sampling(soft_labels, num_labeled_samples)

    # sample the uncertain samples from the validation set
    X_new_val = X_val[uncertain_samples]
    Y_new_weak_val = Y_weak_val[uncertain_samples]

    # add the newly sampled validation data into training set
    X_train = np.concatenate((X_train, X_new_val))
    Y_train = np.concatenate((Y_train, Y_new_weak_val))

    # remove the newly sampled validation data from existing validation set
    X_val = np.delete(X_val, uncertain_samples, axis=0)
    Y_weak_val = np.delete(Y_weak_val, uncertain_samples, axis=0)

    # incrementing the queries made
    queried_samples += uncertain_samples

    # training model with new training set
    classifier, X_train, Y_train, Y_test_predicted, Y_val_predicted = train(X_train, Y_train, X_val, X_test)
    get_test_accuracy(active_iteration, Y_test, Y_test_predicted, classification_accuracies)

print("classification accuracies: ", classification_accuracies)