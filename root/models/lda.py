import pandas as pd
import numpy as np
import math
import random

def cov_matrix(df, binary_feature):
    return np.array(df.drop(binary_feature, axis='columns').cov().values)


# Returns P(y=1)/P(y=0)
def prob_proportion(df, binary_feature):
    p1 = len(df[df[binary_feature] == 1])/len(df)
    p0 = len(df[df[binary_feature] == 0])/len(df)
    return p1/p0


def mean_vector(df, binary_feature, class_val):
    data = df[df[binary_feature] == class_val]
    data = data.drop(binary_feature, axis='columns')
    return np.array(list(data.mean()))


# Returns a function that takes in a vector of feature values
def fit(data, binary_feature):
    # Calculate mean vectors and covariance matrix
    cov = cov_matrix(data, binary_feature)
    mean0 = mean_vector(data, binary_feature, 0)
    mean1 = mean_vector(data, binary_feature, 1)
    p1overp0 = prob_proportion(data, binary_feature)

    # Define the log odds function based on the inputted data set
    def lda_func(param_vector):
        t1 = p1overp0
        t2 = (-1/2)*mean1 @ np.linalg.inv(cov) @ np.transpose(mean1)
        t3 = (1/2)*mean0 @ np.linalg.inv(cov) @ np.transpose(mean0)
        t4 = np.dot(np.dot(param_vector, np.linalg.inv(cov)), np.transpose(np.subtract(mean1, mean0)))
        return math.log(t1, math.e)+t2+t3+t4

    prediction_func = lda_func
    return prediction_func


# Returns a list of predictions on a set given a training set and validation set
def predict(training_set, dataset, binary_feature):
    prediction_func = fit(training_set, binary_feature)
    predictions = []
    for i in range(0, len(dataset)):
        row = dataset.drop(binary_feature, axis='columns')
        param_vector = np.array([dataset.drop(binary_feature, axis='columns').iloc[i]])
        pred = prediction_func(param_vector)
        predictions.append(pred)
    return list(map(lambda x: 1 if x > 0 else 0, predictions))


