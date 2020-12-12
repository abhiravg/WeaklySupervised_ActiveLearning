# WeaklySupervised_ActiveLearning

```
usage: main.py [-h] [--num_labeled_samples NUM_LABELED_SAMPLES]
               [--max_queries MAX_QUERIES] [--dataset {IMDB,YELP,SST-2,TREC}]
               --dataset_home DATASET_HOME
               [--scenario {supervised,weak_supervision,active_learning,active_weak_learning}]
               [--learner {LR,NB,SVC,SGD}]
               [--sample_method {Margin_Sampling,Entropy_Sampling,Random_Sampling}]
               [--label_gen_model {Label_Model,Majority_Voter}]

Active Learning with Weak Supervision - Sentiment Analysis

optional arguments:
  -h, --help            show this help message and exit
  --num_labeled_samples NUM_LABELED_SAMPLES
                        number of initial labeled samples for Active Learning
  --max_queries MAX_QUERIES
                        maximum number of queries for Active Learning
  --dataset {IMDB,YELP,SST-2,TREC}
                        the dataset to run the algorithm on
  --dataset_home DATASET_HOME
                        the dir where dataset resides
  --scenario {supervised,weak_supervision,active_learning,active_weak_learning}
                        the learning algorithm
  --learner {LR,NB,SVC,SGD}
                        the learning algorithm
  --sample_method {Margin_Sampling,Entropy_Sampling,Random_Sampling}
                        the sampling method for Active Learning
  --label_gen_model {Label_Model,Majority_Voter}
                        weak supervision label generation model
```
