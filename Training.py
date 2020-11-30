from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# import numpy as np


def train(x_train, y_train, x_val, x_test):
    classifier = RandomForestClassifier(n_estimators=500, class_weight="balanced")
    classifier.fit(x_train, y_train)
    y_test_predicted = classifier.predict(x_test)
    y_val_predicted = classifier.predict(x_val)
    return classifier, y_test_predicted, y_val_predicted


def get_test_accuracy(iteration, y_test, y_pred, accuracies):
    classification_accuracy = accuracy_score(y_test, y_pred)
    # classification_accuracy = np.mean(y_test_predicted.ravel() == y_test.ravel()) * 100
    accuracies.append(classification_accuracy)
    print("Iteration: ", iteration)
