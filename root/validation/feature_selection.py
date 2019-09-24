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


# Returns the dataset with all unselected features removed
def filter_features(dataset, binary_feature, included_features):
    excluded_features = list(filter(lambda x: x not in included_features + [binary_feature], list(dataset.columns)))
    filtered_data = dataset
    for x in excluded_features:
        filtered_data = filtered_data.drop(x, axis='columns')

    return filtered_data


# Best first expansion of features
def get_best_features(dataset, binary_feature, accuracy_function, max_failures = 50):
    features = list(dataset.columns)
    features.remove(binary_feature)
    best_accuracy = 0
    best_features = []
    included_features = [binary_feature]
    failure_count = 0

    # Repeatedly expand until there is a continuous decrease in accuracy
    while len(included_features) <= len(features):
        print(f'Expanding to {len(included_features)+1} features. Best Accuracy: {best_accuracy}. Failure Count: {failure_count}')
        # Evaluate expansion on all features and choose the best one
        curr_best_acc = 0
        curr_best_feature = None

        for x in features:
            # Find x with greatest accuracy when included
            print(included_features + [x])
            filtered_dataset = filter_features(dataset, binary_feature, included_features + [x])
            x_acc = kfold_validate(filtered_dataset, 5, accuracy_function, binary_feature)
            if x_acc > curr_best_acc:
                curr_best_acc = x_acc
                curr_best_feature = x

        included_features.append(curr_best_feature)  # Include the best feature for this expansion level
        features.remove(curr_best_feature)
        if curr_best_acc > best_accuracy:
            # If the inclusion also beat the best overall accuracy record this feature set as best
            best_accuracy = curr_best_acc
            best_features = list(included_features)
            failure_count = 0
        else:
            # If it didn't, then don't update the bests, mark as a failure and continue
            failure_count += 1
            if failure_count >= max_failures:
                break

    # Either all features have been tried or we consecutive failures exceeded the max
    return best_accuracy, best_features


