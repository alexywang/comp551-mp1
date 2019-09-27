import numpy
import numpy as np
import pandas as pd


def imp(BIN_FEATURE):
    raw_data = pd.read_csv("breast-cancer-wisconsin.csv")
    treated_data = raw_data
    treated_data = (treated_data.loc[treated_data["compactness"] != '?'])
    treated_data.reset_index(drop=True, inplace=True)
    cov_dataset = treated_data.drop(BIN_FEATURE, axis='columns')
    b = (treated_data.loc[treated_data["BorM"] == 2])  # Get all rows that are Benign
    m = (treated_data.loc[treated_data["BorM"] == 4])  # Get all rows that a malignant
    b.reset_index(drop=True, inplace=True)  # Reset the index pos's
    m.reset_index(drop=True, inplace=True)  # Reset the index pos's

    return m.reset_index


def get_proportion(BIN_FEATURE):
    b_rows = b['BorM'].count()  # number of rows where BorM = 2
    m_rows = m['BorM'].count()  # number of rows where BorM = 4
    treated_rows = treated_data['BorM'].count()  # total number of rows
    py_m = m_rows / treated_rows  # proportion of rows that are malignant
    py_b = 1 - py_m
    return py_b

    # b_vector=np.array(b_means)
    # b_mean_array_transpose = np.transpose(b_vector)
    # m_vector=np.array(m_means)
    # m_mean_array_transpose = np.transpose(m_vector)
    # print("\n")
    # drop_binary_feature = treated_data.drop(BIN_FEATURE, axis='columns')


def cov_matrix(df, binary_feature):
    return df.drop(binary_feature, axis='columns').cov().values


def mean_vector(df, binary_feature, class_val):
    data = df[df[binary_feature] == class_val]
    data = data.drop(binary_feature, axis='columns')
    return np.array(data.mean())


def predict(x, dataset, binary_feature):
    meanb =  mean_vector(dataset,binary_feature,2)
    meanm = mean_vector(dataset, binary_feature, 4)
    propb = get_proportion(binary_feature)
    propm = 1-propb
    val1 = propb / propm
    val2 = ((1 / 2) * numpy.matrix.transpose(((meanm))*(numpy.linalg.inv(cov_matrix(dataset, binary_feature)))))*(meanm)
    val3 = ((1 / 2) * numpy.matrix.transpose(((meanb))*(numpy.linalg.inv(cov_matrix(dataset, binary_feature)))))*(meanb)
    val4 = numpy.matrix.transpose(x)*(numpy.linalg.inv(cov_matrix(dataset, binary_feature)))*(meanb-meanm)
    return val1 - val2 + val3 + val4

    print(predict())
