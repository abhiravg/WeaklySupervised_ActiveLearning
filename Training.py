from sklearn.ensemble import RandomForestClassifier
import numpy as np


def train(x_train, y_train, x_val, x_test):
    classifier = RandomForestClassifier(n_estimators=500, class_weight="balanced")
    classifier.fit(x_train, y_train)
    y_test_predicted = classifier.predict(x_test)
    y_val_predicted = classifier.predict(x_val)
    return classifier, x_train, y_train, y_test_predicted, y_val_predicted


def get_test_accuracy(iteration, y_test, y_test_predicted, accuracies):
    classification_accuracy = np.mean(y_test_predicted.ravel() == y_test.ravel()) * 100
    accuracies.append(classification_accuracy)
    print("Iteration: ", iteration)
