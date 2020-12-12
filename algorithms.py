from preprocessing import fetch_and_preprocess_data, split_data
from Training import train, get_test_accuracy
from weak_supervision import weak_supervisor
from active_learning import active_learner


def supervised_learning(dataset, dataset_home, learning_model):
    train_df, test_df = fetch_and_preprocess_data(dataset, dataset_home)
    X_train, Y_train, X_test, Y_test = split_data(train_df, test_df)

    classifier, Y_test_predicted = train(X_train, Y_train, X_test, learning_model)
    metrics = get_test_accuracy(1, Y_test, Y_test_predicted)
    return metrics

def weak_supervision_learning(dataset, dataset_home, learning_model, weak_label_model):
    train_df, test_df, unlab_df = fetch_and_preprocess_data(dataset, dataset_home, unlab=True)
    unlab_df = weak_supervisor(unlab_df, weak_label_model)
    unlab_df.rename(columns={"weak_labels": "target"}, inplace=True)

    X_train, Y_train, X_test, Y_test = split_data(train_df, test_df, 
                                        unlab_df=unlab_df, method="weak_supervision")
    
    classifier, Y_test_predicted = train(X_train, Y_train, X_test, learning_model)
    metrics = get_test_accuracy(1, Y_test, Y_test_predicted)
    return metrics

def active_learning(dataset, dataset_home, learning_model, sample_selection_method,
    num_labeled_samples, max_queries, weak_supervision=False, weak_label_model=None):
    if weak_supervision:
        return active_learner(num_labeled_samples, max_queries, learning_model,
        sample_selection_method, dataset, dataset_home, label_method=weak_label_model, weak_supervision=True)
    else:
        return active_learner(num_labeled_samples, max_queries, learning_model,
        sample_selection_method, dataset, dataset_home)