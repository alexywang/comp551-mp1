import pandas as pd
import numpy as numpy


# Temp import code for testing...
def get_data():
    """
        reads data from breast cancer data document. Adds data to treated_data, finds the last column which
        indicates Malignant or Benign and drops it.
    """
    raw_data = pd.read_csv("../data/cancer/breast-cancer-wisconsin.data", sep=",")
    treated_data = raw_data[raw_data.columns[:-1]]
    return treated_data


# partition a dataframe into k parts
def partition(dataframe, k):
    partition_size = len(dataframe) // k
    partitions = []
    for i in range(0, k - 1):
        partitions.append(dataframe[i * partition_size: (i + 1) * partition_size])

    partitions.append(dataframe[k - 1 * partition_size:])
    return partitions


# validates a model using k-fold validation, and returns the average error across all validation set configurations
def kfold_validate(dataframe, k, train_and_predict, binary_feature):
    partitions = partition(dataframe, k)

    # Hold out each set for training once and average resulting weights
    accuracies = []
    for holdout in partitions:
        training_data = [x for x in partitions if x != holdout]
        # TODO: train model on training set, then check accuracy with holdout
        accuracies.append(train_and_predict(training_data, holdout, binary_feature))

    # Average out and return the accuracies for the given features
    return sum(accuracies)/len(accuracies)




