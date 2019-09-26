import pandas as pd


def get_data():
    """
        reads data from breast cancer data document. Adds data to treated_data, finds the last column which
        indicates Malignant or Benign and drops it.
    """
    raw_data = pd.read_csv("breast-cancer-wisconsin.csv", sep=";")
    treated_data = raw_data
    columns = treated_data.columns.tolist()  # get the columns
    cols_to_use = columns[:len(columns) - 1]  # drop the last one
    treated_data = read_csv(path, usecols=cols_to_use)
    return treated_data

