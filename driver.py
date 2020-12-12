from algorithms import supervised_learning, weak_supervision_learning, active_learning


def evaluate(args):
  # initial labeled samples
  num_initial_labeled_samples = args.num_labeled_samples

  # maximum number of queries
  max_queries = args.max_queries

  # dataset
  dataset = args.dataset
  dataset_home = args.dataset_home

  # learning_type
  learning_type = args.scenario

  # machine learning model
  learning_model = args.learner

  # sample selection method
  sample_selection_method = args.sample_method

  # weak label generation model
  weak_label_generation_model = args.label_gen_model

  if learning_type == "supervised":
    print("Scenario: Supervised Learning")
    return supervised_learning(dataset, dataset_home, learning_model)
  elif learning_type == "weak_supervision":
    print("Scenario: Weak Supervision")
    return weak_supervision_learning(dataset, dataset_home, learning_model, weak_label_generation_model)
  elif learning_type == "active_learning":
    print("Scenario: Active Learning")
    return active_learning(dataset, dataset_home, learning_model, sample_selection_method,
            num_initial_labeled_samples, max_queries)
  else:
    print("Scenario: Active Learning with Weak Supervision")
    return active_learning(dataset, dataset_home, learning_model, sample_selection_method,
            num_initial_labeled_samples, max_queries, weak_supervision=True, weak_label_model=weak_label_generation_model)
