import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel, MajorityLabelVoter

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


def weak_supervisor(dataframe, model_type):
    labeling_functions = [positive_labeling_function, positive1_labeling_function, negative_labeling_function,
                          negative1_labeling_function]
    pandasApplier = PandasLFApplier(lfs=labeling_functions)
    label_training_matrix = pandasApplier.apply(df=dataframe)

    if model_type == "label_model":
        # constructing a probabilistic label model
        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_train=label_training_matrix, n_epochs=300, log_freq=50, seed=123)
        dataframe["weak_labels"] = label_model.predict(L=label_training_matrix)
        print("dataframe shape: ", dataframe.shape)
        dataframe = dataframe[dataframe["weak_labels"] != -1]
        print("dataframe shape after filtering: ", dataframe.shape)
        return dataframe

    else:
        majorityLabelVoter = MajorityLabelVoter()
        dataframe["weak_labels"] = majorityLabelVoter.predict(L=label_training_matrix)
        print("dataframe shape: ", dataframe.shape)
        dataframe = dataframe[dataframe["weak_labels"] != -1]
        print("dataframe shape after filtering: ", dataframe.shape)
        return dataframe



