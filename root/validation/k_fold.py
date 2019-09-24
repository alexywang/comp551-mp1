import pandas as pd
import numpy as numpy


# partition a dataframe into k parts
def partition(dataframe, k):
    partition_size = len(dataframe) // k
    partitions = []
    for i in range(0, k - 1):
        partitions.append(dataframe.iloc[i * partition_size: (i + 1) * partition_size])

    partitions.append(dataframe.iloc[k - 1 * partition_size:])
    return partitions


# validates a model using k-fold validation, and returns the average error across all validation set configurations
def kfold_validate(dataframe, k, train_and_predict, binary_feature):
    partitions = partition(dataframe, k)

    # Hold out each set for training once and average resulting weights
    accuracies = []
    iteration = 0
    for i in range(0, len(partitions)):
        holdout = partitions[i]
        training_data = pd.DataFrame(columns=partitions[0].columns)
        for j in range(0, len(partitions)):
            if j != i:
                training_data.append(partitions[j])
        accuracies.append(train_and_predict(training_data, holdout, binary_feature))

    # Average out and return the accuracies for the given features
    return sum(accuracies)/len(accuracies)






