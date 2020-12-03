from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
import numpy as np


def train(x_train, y_train, x_test, model_type):
    accuracy_scores = []
    # model type
    if model_type == "RandomForest":
        classifier = RandomForestClassifier(n_estimators=500, class_weight="balanced")
    elif model_type == "SVC":
        classifier = SVC(C=1, kernel="linear", probability=True, class_weight="balanced")
    else:
        classifier = MultinomialNB()
    # stratified 10-fold cross validation
    cross_validation = KFold(n_splits=5, random_state=42, shuffle=False)
    index = 1
    for train_index, test_index in cross_validation.split(x_train):
        print("Training Iteration: ", index)
        x_training, x_validation, y_training, y_validation = \
            x_train[train_index], x_train[test_index], y_train[train_index], y_train[test_index]

        classifier.fit(x_training, y_training)
        accuracy_scores.append(classifier.score(x_validation, y_validation))

        index += 1
    y_test_predicted = classifier.predict(x_test)
    print("cross_validation training accuracy: ", np.mean(accuracy_scores))
    return classifier, y_test_predicted


def get_test_accuracy(iteration, y_test, y_pred, accuracies):
    # dictionary = metrics.classification_report(y_test, y_pred)
    classification_accuracy = accuracy_score(y_test, y_pred)
    # classification_accuracy = np.mean(y_test_predicted.ravel() == y_test.ravel()) * 100
    accuracies.append(classification_accuracy)
    print("classification accuracy: ", classification_accuracy)
    print("Iteration: ", iteration)
    # print("classification_report: ", dictionary)
