import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from snorkel.preprocess import preprocessor
import numpy as np
from textblob import TextBlob

# ABSTAIN = -1
# positive = 1
# negative = 0

@labeling_function()
def positive_labeling_function(text):
    positive_signals = ["good", "wonderful", "amazing"]
    data = text["data"].split(" ")
    for positive_signal in positive_signals:
        if positive_signal in data:
            return 1

    return -1


@labeling_function()
def positive1_labeling_function(text):
    positive_signals1 = ["excellent", "great"]
    data = text["data"].split(" ")
    for positive_signal1 in positive_signals1:
        if positive_signal1 in data:
            return 1

    return -1

@labeling_function()
def negative_labeling_function(text):
    negative_signals = ["bad", "horrible", "sucks"]
    data = text["data"].split(" ")
    for negative_signal in negative_signals:
        if negative_signal in data:
            return 0

    return -1


@labeling_function()
def negative1_labeling_function(text):
    negative_signals1 = ["awful", "terrible"]
    data = text["data"].split(" ")
    for negative_signal1 in negative_signals1:
        if negative_signal1 in data:
            return 0

    return -1


@preprocessor(memoize=True)
def textblob_sentiment_analyzer(text):
    sentiment_scores = TextBlob(text['data'])
    text['polarity'] = sentiment_scores.sentiment.polarity
    text['subjectivity'] = sentiment_scores.sentiment.subjectivity
    return text


@labeling_function(pre=[textblob_sentiment_analyzer])
def textblob_positive_sentiment(text):
    if text['polarity'] >= 0.1:
        return 1
    else:
        return -1


@labeling_function(pre=[textblob_sentiment_analyzer])
def textblob_negative_sentiment(text):
    if text['polarity'] <= 0:
        return 0
    else:
        return -1


def weak_supervisor(dataframe, model_type):
    # if labeling_function_type == "pre_trained_model":
    #     labeling_functions = [textblob_positive_sentiment, textblob_negative_sentiment]
    # else:
    labeling_functions = [positive_labeling_function, positive1_labeling_function, negative_labeling_function,
                              negative1_labeling_function, textblob_positive_sentiment, textblob_negative_sentiment]
    pandasApplier = PandasLFApplier(lfs=labeling_functions)
    label_training_matrix = pandasApplier.apply(df=dataframe)

    if model_type == "Label_Model":
        print("Weak Supervision Model: Label_Model")
        # constructing a probabilistic label model
        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_train=label_training_matrix, n_epochs=300, log_freq=50, seed=123)
        dataframe["weak_labels"] = label_model.predict(L=label_training_matrix)
        # print("dataframe shape: ", dataframe.shape)
        dataframe = dataframe[dataframe["weak_labels"] != -1]
        # verify_weak_signals(dataframe)
        # print("dataframe shape after filtering: ", dataframe.shape)
        return dataframe

    else:
        print("Weak Supervision Model: Majority_Voter")
        majorityLabelVoter = MajorityLabelVoter()
        prob = majorityLabelVoter.predict_proba(label_training_matrix)
        # print("predict_prob: ", prob[0:20, :])
        dataframe["weak_labels"] = majorityLabelVoter.predict(L=label_training_matrix)
        # print("df weak labels: ", dataframe[0:20]['weak_labels'])
        # print("dataframe shape: ", dataframe.shape)
        dataframe = dataframe[dataframe["weak_labels"] != -1]
        # verify_weak_signals(dataframe)
        # print("dataframe shape after filtering: ", dataframe.shape)
        return dataframe


def verify_weak_signals(dataframe):
    accuracy = np.mean((dataframe['target'].ravel() == dataframe['weak_labels'].ravel()))*100
    print("Weak signals accuracy: ", accuracy)


