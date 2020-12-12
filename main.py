#!/usr/bin/env python

from pprint import pprint

from driver import evaluate

import argparse

parser = argparse.ArgumentParser(
    description="Active Learning with Weak Supervision - Sentiment Analysis")

parser.add_argument("--num_labeled_samples", type=int, default=2000,
                help="number of initial labeled samples for Active Learning")
parser.add_argument("--max_queries", type=int, default=20000,
                help="maximum number of queries for Active Learning")
parser.add_argument("--dataset", type=str, default="IMDB",
                choices=["IMDB", "YELP", "SST-2", "TREC"],
                help="the dataset to run the algorithm on")
parser.add_argument("--dataset_home", type=str, required=True,
                help="the dir where dataset resides")
parser.add_argument("--scenario", type=str, default="active_weak_learning",
                choices=["supervised", "weak_supervision", "active_learning",
                "active_weak_learning"], help="the learning algorithm")
parser.add_argument("--learner", type=str, default="LR", choices=["LR", "NB", "SVC", "SGD"],
                help="the learning algorithm")
parser.add_argument("--sample_method", type=str, default="Margin_Sampling",
                choices=["Margin_Sampling", "Entropy_Sampling", "Random_Sampling"],
                help="the sampling method for Active Learning")
parser.add_argument("--label_gen_model", type=str, default="Label_Model",
                choices=["Label_Model", "Majority_Voter"],
                help="weak supervision label generation model")

args = parser.parse_args()

if args.max_queries < args.num_labeled_samples:
    raise ValueError("ERROR: Initial labeled samples cannot be greater than max queries.")

print(args)
results = evaluate(args)

# Graphing or any other visualizations
pprint(results)
