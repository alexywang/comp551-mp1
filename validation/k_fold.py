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
    partitionSize = len(dataframe) // k
    partitions = []
    for i in range(0, k - 1):
        partitions.append(dataframe[i * partitionSize: (i + 1) * partitionSize])

    partitions.append(dataframe[k - 1 * partitionSize:])
    return partitions

# validates a model using k-fold validation, and returns the average error across all configurations
def kFoldValidation(dataframe, k, descent = 0):
    partitions = partition(dataframe, k)

    # Hold out each set for training once and average resulting weights
    results = []
    for holdout in partitions:
        trainingSet = [x for x in partitions if x != holdout]
        # TODO: train model on training set, then check accuracy with holdout

    # TODO: Average out accuracy and return final evaluation of the model




