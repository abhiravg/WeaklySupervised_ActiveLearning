from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import metrics

import numpy as np


# training with cross validation        
def train(x_train, y_train, x_test, model_type, scenario="supervised"):
    accuracy_scores = []
    # model type
    if model_type == "SGD":
        print("Model: Linear SGD")
        classifier = SGDClassifier(class_weight="balanced", random_state=123)
    elif model_type == "SVC":
        print("Model: Linear SVC")
        classifier = LinearSVC(class_weight="balanced", random_state=123)
    elif model_type == "LR":
        print("Model: Logistic Regression")
        classifier = LogisticRegression(class_weight="balanced", solver="lbfgs", random_state=123)
    else:
        print("Model: Naive Bayes")
        classifier = MultinomialNB()

    if scenario == "supervised":
        # stratified 10-fold cross validation
        cross_validation = KFold(n_splits=10, random_state=42, shuffle=False)
        index = 1
        for train_index, test_index in cross_validation.split(x_train):
            print("Training Iteration: ", index)
            x_training, x_validation, y_training, y_validation = \
                x_train[train_index], x_train[test_index], y_train[train_index], y_train[test_index]

            classifier.fit(x_training, y_training)
            accuracy_scores.append(classifier.score(x_validation, y_validation))

            index += 1
        print("Cross validation training accuracy: ", np.mean(accuracy_scores))        
    else:
        classifier.fit(x_train, y_train)
    
    # prediction on test data
    y_test_predicted = classifier.predict(x_test)

    return classifier, y_test_predicted


def get_test_accuracy(iteration, y_test, y_pred):
    print("Iteration: ", iteration)
    print("Test accuracy: ", metrics.accuracy_score(y_test, y_pred))
    classification_metrics = metrics.classification_report(y_test, y_pred, output_dict=True)
    return classification_metrics
