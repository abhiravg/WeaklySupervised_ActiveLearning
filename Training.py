from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import pprint
import numpy as np


# training without cross validation
def active_learning_train(x_train, y_train, x_test, model_type):
    # model type
    if model_type == "RandomForest":
        classifier = RandomForestClassifier(n_estimators=500, class_weight="balanced")
    elif model_type == "SVC":
        classifier = SVC(C=1, kernel="linear", probability=True, class_weight="balanced")
    elif model_type == "Logistic":
        classifier = LogisticRegression(class_weight="balanced")
    else:
        classifier = MultinomialNB()
    
    classifier.fit(x_train, y_train)
    
    # prediction on test data
    y_test_prediction = classifier.predict(x_test)
    
    return classifier, y_test_prediction
        
        
# training with cross validation        
def supervised_learning_train(x_train, y_train, x_test, model_type):
    accuracy_scores = []
    # model type
    if model_type == "Random_Forest":
        classifier = RandomForestClassifier(n_estimators=500, class_weight="balanced")
    elif model_type == "SVC":
        classifier = SVC(C=1, kernel="linear", probability=True, class_weight="balanced")
    elif model_type == "Logistic_Regression":
        classifier = LogisticRegression(class_weight="balanced")
    else:
        classifier = MultinomialNB()
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
    y_test_predicted = classifier.predict(x_test)
    print("cross_validation training accuracy: ", np.mean(accuracy_scores))
    return classifier, y_test_predicted


def get_test_accuracy(iteration, y_test, y_pred):
    print("Iteration: ", iteration)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    classification_metrics = metrics.classification_report(y_test, y_pred, output_dict=True)
    return classification_metrics
