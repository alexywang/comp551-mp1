import numpy
import numpy as np
import pandas as pd
import math


# def imp(BIN_FEATURE):
#     raw_data = pd.read_csv("breast-cancer-wisconsin.csv")
#     treated_data = raw_data
#     treated_data = (treated_data.loc[treated_data["compactness"] != '?'])
#     treated_data.reset_index(drop=True, inplace=True)
#     cov_dataset = treated_data.drop(BIN_FEATURE, axis='columns')
#     b = (treated_data.loc[treated_data["BorM"] == 2])  # Get all rows that are Benign
#     m = (treated_data.loc[treated_data["BorM"] == 4])  # Get all rows that a malignant
#     b.reset_index(drop=True, inplace=True)  # Reset the index pos's
#     m.reset_index(drop=True, inplace=True)  # Reset the index pos's
#
#     return m.reset_index

def get_data():
    """
        reads data from breast cancer data document. Adds data to treated_data, finds the last column which
        indicates Malignant or Benign and drops it.
    """
    data = pd.read_csv("breast-cancer-wisconsin.csv", sep=',')
    # treated_data = (raw_data[raw_data.columns[:-1]])
    # treated_data = (treated_data.loc[treated_data["compactness"] != '?'])
    # treated_data.reset_index(drop=True, inplace=True)
    data = data[data['bare_nuclei'] != '?']
    data = data.drop('id', axis='columns')
    data['class'] = data['class'].apply(lambda x: 0 if x == 2 else 1)
    data = data.sample(n=len(data), random_state=42).reset_index(drop=True)
    data = data.astype({'bare_nuclei': 'int64'})
    return data


# Returns P(y=1)/P(y=0)
def get_proportion(df, binary_feature):
    p1 = len(df[df[binary_feature] == 1])/len(df)
    p0 = len(df[df[binary_feature] == 0])/len(df)
    return p1/p0


def cov_matrix(df, binary_feature):
    return np.array(df.drop(binary_feature, axis='columns').cov().values)


def mean_vector(df, binary_feature, class_val):
    data = df[df[binary_feature] == class_val]
    data = data.drop(binary_feature, axis='columns')
    return np.array([data.mean()])


def predict(x, dataset, binary_feature):
    meanb = mean_vector(dataset, binary_feature, 0)
    meanm = mean_vector(dataset, binary_feature, 1)
    cov = cov_matrix(dataset, binary_feature)
    propb = get_proportion(dataset, binary_feature)
    propm = 1-propb
    val1 = get_proportion(dataset, binary_feature)
    val2 = -(1 / 2) * numpy.dot(numpy.dot(meanm, numpy.linalg.inv(cov)), numpy.transpose(meanm))
    val3 = (1 / 2) * numpy.dot(numpy.dot(meanb, numpy.linalg.inv(cov)), numpy.transpose(meanb))
    val4 = numpy.dot(numpy.dot(x, numpy.linalg.inv(cov)), numpy.transpose(numpy.subtract(meanm, meanb)))
    print(val1, val2, val3, val4)
    print(np.array(val1+val2+val3+val4)[0][0])
    return math.log(np.array(val1 + val2 + val3 + val4)[0][0], math.e)


data = get_data()
xv = None
xv= data.drop('class', axis='columns').iloc[0]
# for rows in data:
#     x = data[rows][0]
#     break
xv = np.array([xv])

predict(xv, data, 'class')
