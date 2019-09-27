import pandas as pd
import numpy as np
import math


def get_data():
    """
        reads data from breast cancer data document. Adds data to treated_data, finds the last column which
        indicates Malignant or Benign and drops it.
    """
    data = pd.read_csv("../breast-cancer-wisconsin.csv", sep=',')
    # treated_data = (raw_data[raw_data.columns[:-1]])
    # treated_data = (treated_data.loc[treated_data["compactness"] != '?'])
    # treated_data.reset_index(drop=True, inplace=True)
    data = data[data['bare_nuclei'] != '?']
    data = data.drop('id', axis='columns')
    data['class'] = data['class'].apply(lambda x: 0 if x == 2 else 1)
    data = data.sample(n=len(data), random_state=42).reset_index(drop=True)
    data = data.astype({'bare_nuclei': 'int64'})
    return data


def cov_matrix(df, binary_feature):
    return np.array([df.drop(binary_feature, axis='columns').cov().values])


# Returns P(y=1)/P(y=0)
def prob_proportion(df, binary_feature):
    p1 = len(df[df[binary_feature] == 1])/len(df)
    p0 = len(df[df[binary_feature] == 0])/len(df)
    return p1/p0


def mean_vector(df, binary_feature, class_val):
    data = df[df[binary_feature] == class_val]
    data = data.drop(binary_feature, axis='columns')
    return np.array([data.mean()])


# Returns a function that takes in a vector of feature values
def fit(training_set, binary_feature):
    cov = cov_matrix(training_set, binary_feature)
    mean0 = mean_vector(training_set, binary_feature, 0)
    mean1 = mean_vector(training_set, binary_feature, 1)
    # def lda_func(param_vector):


d = get_data()

a = np.array([[1,2,3,4,5]])

print(np.transpose(cov_matrix(d, 'class')))
print()
print(cov_matrix(d, 'class'))