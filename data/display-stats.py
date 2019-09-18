import numpy as np
import scipy.stats as st
import collections

''' Analysis of the data properties. For each feature, we display:
    - Mean
    -Std. Dev
    -Excess kurtosis (dispersion measure focused on tail sizes)

    For categorical data, we display proportion information.
'''


def display_stats(treated_data, categorical_features):
    for feature in treated_data.keys():
        values = treated_data[feature]
        if feature not in categorical_features:
            print(feature + ":")
            print("Mean : ", np.mean(values))
            print("Stdev : ", np.std(values))
            print("Excess kurtosis : ", st.kurtosis(values))
            print("")

        else:
            print(feature + ":")
            print(collections.Counter(values))
            print("")
