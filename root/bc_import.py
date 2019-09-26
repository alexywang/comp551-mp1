import pandas as pd


def get_data():
    """
        reads data from breast cancer data document. Adds data to treated_data, finds the last column which
        indicates Malignant or Benign and drops it.
    """
    raw_data = pd.read_csv("breast-cancer-wisconsin.csv")
    treated_data = (raw_data[raw_data.columns[:-1]])
    treated_data = (treated_data.loc[treated_data["compactness"] != '?'])
    treated_data.reset_index(drop=True, inplace=True)
    return treated_data


print(get_data())
