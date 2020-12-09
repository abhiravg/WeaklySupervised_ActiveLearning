import sys
from evaluate import active_weakly_supervised_learning
from supervised_learning import supervised_learn
from snorkel_weakSupervision import weak_supervision
import pandas as pd
import pprint

if len(sys.argv) < 6:
    print("ERROR: Missing command line arguments")
    sys.exit(1)

# initial labeled samples
num_initial_labeled_samples = int(sys.argv[1])

# maximum number of queries
max_num_queries = int(sys.argv[2])

# dataset
dataset = str(sys.argv[3])
dataset_home = str(sys.argv[5])

# learning_type: supervised, weak_supervision, active_weak_learning, active_learning
learning_type = str(sys.argv[4])


if max_num_queries < num_initial_labeled_samples:
    print("ERROR: initial labeled examples(arg 1) cannot be greater than max_num_queries (arg 2)")

# machine learning models
learning_models = ["Naive_Bayes", "Logistic_Regression"]#, "Random_Forest"]
# learning_models = ["Naive_Bayes"]
# learning_models = ["Logistic_Regression"]

# sample selection methods
sample_selection_methods = ["Random_Sampling", "Margin_Sampling", "Entropy_Sampling"]


# weak label generation models
# weak_label_generation_models = ["Label_Model", "Majority_Voter"]
weak_label_generation_models = ["Label_Model"]

if learning_type == "supervised":
    supervised_learning_benchmark = {}
    print("Evaluating supervised learning...")
    for learning_model in learning_models:
        print("Learning Model: ", learning_model)
        supervised_learning_benchmark[learning_model] = supervised_learn(
            f"{dataset_home}/train", f"{dataset_home}/test", learning_model)
    
    pprint.pprint(supervised_learning_benchmark, width=1)

elif learning_type == "weak_supervision":
    weak_supervision_benchmark = {}
    for learning_model in learning_models:
        weak_supervision_benchmark[learning_model] = {}
        for weak_label_generation_model in weak_label_generation_models:
            weak_supervision_benchmark[learning_model][weak_label_generation_model] = \
                weak_supervision(f"{dataset_home}/train", f"{dataset_home}/test", weak_label_generation_model, learning_model)
    
    with open('result_ws.out', 'w') as f:
        pprint.pprint(weak_supervision_benchmark, f)

else:
    active_learning_benchmarks = {}
    print("Evaluating Active Learning...")
    for learning_model in learning_models:
        active_learning_benchmarks[learning_model] = {}
        print("Learning Model: ", learning_model)
        for sample_selection_method in sample_selection_methods:
            active_learning_benchmarks[learning_model][sample_selection_method] = {}
            print("sample selection method: ", sample_selection_method)
            for weak_label_generation_model in weak_label_generation_models:
                print("label_generation_model: ", weak_label_generation_model)
                active_learning_benchmarks[learning_model][sample_selection_method][weak_label_generation_model] = \
                    active_weakly_supervised_learning(num_initial_labeled_samples, max_num_queries, learning_model,
                                                      sample_selection_method, weak_label_generation_model, dataset, dataset_home)
                
    print("Active learning benchmarks......")
    with open('result.out', 'w') as f:
        pprint.pprint(active_learning_benchmarks, f)
