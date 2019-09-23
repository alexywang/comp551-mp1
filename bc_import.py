import dataframe as dataframe
import pandas as pd

def get_data():
    """
        reads data from breast cancer data document. Adds data to treated_data, finds the last column which
        indicates Malignant or Benign and drops it.
    """
    raw_data = pd.read_csv("breast-cancer-wisconsin.csv", sep=",")
    treated_data = raw_data[raw_data.columns[:-1]]
    return treated_data


with open("breast-cancer-wisconsin.csv", "r") as f:
    lines = f.readlines()
 with open("breast-cancer-wisconsin.csv", "w") as f:
     for line in lines:
        if line.strip("\n") != "?":
             f.write(line)
# dataframe = dataframe[(dataframe.astype(str) != "?").all(axis=1)]

print (dataframe)
print(get_data())
