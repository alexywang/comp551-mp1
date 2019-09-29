import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wine_import as importer

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

data = importer.get_data_raw('winequality-red.csv')

def plot_matrix():
    pd.plotting.scatter_matrix(data, alpha=0.025, figsize=(30, 30), diagonal='kde')
    plt.show()


def plot_corr():
    correlation = data.corr()
    plt.figure(figsize=(8, 8))
    heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")

def plot_cov():
    cov = data.drop('class', axis='columns').var()
    plt.figure(figsize=(8, 8))
    heatmap = sns.heatmap(cov, annot=True, linewidths=0, vmin=0, cmap="RdBu_r")

def plot_histogram():
    hist = data.hist(bins=10)


plot_histogram()
plt.show()