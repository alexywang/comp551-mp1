import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import

data = pd.read_csv('winequality-red.csv', sep=";")


def plot_matrix():
    pd.plotting.scatter_matrix(data, alpha=0.3, figsize=(30, 30), diagonal='kde')
    plt.show()


def plot_corr():
    correlation = data.corr()
    plt.figure(figsize=(8, 8))
    heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")



plot_matrix()
plt.show()