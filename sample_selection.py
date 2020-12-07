from sklearn.utils import check_random_state
import numpy as np


def sampling_method(label_prob_values, num_labeled_samples, sampling_method):
    if sampling_method == "Random_Sampling":
        samples = random_sampling(label_prob_values, num_labeled_samples)
    elif sampling_method == "Margin_Sampling":
        samples = margin_selection_sampling(label_prob_values, num_labeled_samples)
    else:
        samples = entropy_selection_sampling(label_prob_values, num_labeled_samples)
    
    return samples
        
        
def random_sampling(label_prob_values, num_labeled_samples):
    random_state = check_random_state(0)
    validation_size = label_prob_values.shape[0]
    random_samples = np.random.choice(validation_size, num_labeled_samples, replace=False)
    return random_samples


def margin_selection_sampling(label_prob_values, num_labeled_samples):
    rev = np.sort(label_prob_values, axis=1)[:, ::-1]
    values = rev[:, 0] - rev[:, 1]
    samples = np.argsort(values)[:num_labeled_samples]
    return samples


def entropy_selection_sampling(label_prob_values, num_labeled_samples):
    entropy = (-label_prob_values * np.log2(label_prob_values)).sum(axis=1)
    samples = (np.argsort(entropy)[::-1])[:num_labeled_samples]
    return samples


