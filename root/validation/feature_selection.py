import pandas as pd
import math
import scipy as sp
import numpy as np


# partition a dataframe into k parts
def partition(dataframe, k):
    partition_size = len(dataframe) // k
    partitions = []
    for i in range(0, k - 1):
        partitions.append(dataframe.iloc[i * partition_size: (i + 1) * partition_size])

    partitions.append(dataframe.iloc[k - 1 * partition_size:])
    return partitions


# validates a model using k-fold validation, and returns the average error across all validation set configurations
def evaluate_acc(dataframe, train_and_predict, binary_feature):
    partitions = partition(dataframe, 5)

    # Hold out each set for training once
    accuracies = []
    for i in range(0, len(partitions)):
        holdout = partitions[i]
        training_data = pd.DataFrame(partitions[0])[0:0]
        for j in range(0, len(partitions)):
            if j != i:
                training_data = training_data.append(partitions[j], ignore_index=True)
        accuracies.append(train_and_predict(training_data, holdout, binary_feature)[2])

    # Average out and return the accuracies for the given features
    return sum(accuracies)/len(accuracies)


# Returns a dataframe with correlated features dropped
def remove_correlated(dataset, binary_feature, threshold = 0.8):
    corr_matrix = dataset.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return dataset.drop(dataset[to_drop], axis=1)


# Returns a dataframe with only features that have a variance that exceeds the threshold.
def variance_threshold(dataset, binary_feature, threshold=0.005):
    var_series = dataset.var()
    to_drop = []
    for key in var_series.keys():
        if key != binary_feature and var_series.get(key=key) < 0.005:
            to_drop.append(key)
    dropped = dataset
    for col in to_drop:
        dropped = dropped.drop(col, axis='columns')
    return dropped


# Returns the dataset with all unselected features removed
def filter_features(dataset, binary_feature, included_features):
    excluded_features = list(filter(lambda x: x not in included_features + [binary_feature], list(dataset.columns)))
    filtered_data = dataset
    for x in excluded_features:
        filtered_data = filtered_data.drop(x, axis='columns')

    return filtered_data


def backwards_elimination(dataset, binary_feature, accuracy_function, min_improvement=0):
    included_features = list(dataset.columns)
    candidates = list(dataset.columns)
    candidates.remove(binary_feature)
    candidates.remove('bias')
    print('Evaluating full feature set...')
    best_accuracy = evaluate_acc(dataset, accuracy_function, binary_feature)
    best_features = list(included_features)

    while len(candidates) != 0:
        print(f'Current set size: {len(included_features)-2}. Best Accuracy: {best_accuracy}. Candidates Remaining: {len(candidates)}')
        curr_best_acc = 0
        curr_best_candidate = None
        for x in candidates:
            # Find x with the greatest accuracy when excluded
            filtered_features = filter_features(dataset, binary_feature, [y for y in included_features if y != x])
            x_acc = evaluate_acc(filtered_features, accuracy_function, binary_feature)
            if x_acc > curr_best_acc:
                curr_best_acc = x_acc
                curr_best_candidate = x
            print([y for y in included_features if y != x], f'Accuracy: {x_acc} when removing {x}')

        included_features.remove(curr_best_candidate)
        candidates.remove(curr_best_candidate)
        print(f'Accuracy Improvement = {curr_best_acc - best_accuracy}, Best Candidate: {curr_best_candidate}')
        if curr_best_acc > (best_accuracy + min_improvement):
            best_accuracy = curr_best_acc
            best_features = list(included_features)
        else:
            if min_improvement != 0:
                break
    return best_accuracy, best_features


# Forward search expansion of features
def forward_search(dataset, binary_feature, accuracy_function, min_improvement=0):
    features = list(dataset.columns)
    features.remove(binary_feature)
    features.remove('bias')
    best_accuracy = 0
    best_features = []
    included_features = [binary_feature, 'bias']

    # Repeatedly expand until there is a continuous decrease in accuracy or if we run out of features
    while len(features) != 0:
        print(f'Expanding to {len(included_features)+1} features. Best Accuracy: {best_accuracy}. Features Remaining: {len(features)}')
        # Evaluate expansion on all features and choose the best one
        curr_best_acc = 0
        curr_best_feature = None

        for x in features:
            # Find x with greatest accuracy when included
            filtered_dataset = filter_features(dataset, binary_feature, included_features + [x])
            x_acc = evaluate_acc(filtered_dataset, accuracy_function, binary_feature)
            if x_acc > curr_best_acc:
                curr_best_acc = x_acc
                curr_best_feature = x
            print(included_features + [x], f'Accuracy: {x_acc}')

        included_features.append(curr_best_feature)  # Include the best feature for this expansion level
        features.remove(curr_best_feature)
        print(f'Accuracy Improvement: {curr_best_acc - best_accuracy}')
        if curr_best_acc > (best_accuracy + min_improvement):
            # If the inclusion also beat the best overall accuracy record this feature set as best
            best_accuracy = curr_best_acc
            best_features = list(included_features)
        else:
            if min_improvement != 0:
                break

    # Either all features have been tried or we consecutive failures exceeded the max
    return best_accuracy, best_features



