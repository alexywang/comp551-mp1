import pandas as pd


def get_data(path):
    """
        reads data from breast cancer data document. Adds data to treated_data, finds the last column which
        indicates Malignant or Benign and drops it.
    """
    data = pd.read_csv(path, sep=',')
    # treated_data = (raw_data[raw_data.columns[:-1]])
    # treated_data = (treated_data.loc[treated_data["compactness"] != '?'])
    # treated_data.reset_index(drop=True, inplace=True)
    data = data[data['bare_nuclei'] != '?']
    data = data.drop('id', axis='columns')
    data['class'] = data['class'].apply(lambda x: 0 if x == 2 else 1)
    data = data.sample(n=len(data), random_state=42).reset_index(drop=True)
    data = data.astype({'bare_nuclei': 'int64'})
    return data


