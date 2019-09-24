import numpy as np
import scipy.stats as st
import collections
import itertools


def normalize(dataframe, categorical_features):
    """
        Normalizing the features in the given dataset so that they have
        mean of 0 and stdev of 1. Used to ensure better performance during
        gradient descent, as our features have vastly different scales
        (esp. for the wine data).
    """
    for feature in dataframe.keys():
        values = dataframe[feature]

        if feature not in categorical_features:
            mean = np.mean(values)
            stdev = np.std(values)

            dataframe[feature] = values.apply(lambda x: (x - mean) / stdev)
    return dataframe


def add_bias(dataframe):
    """
        Adds a column filled with only ones with the name "bias". Used to provide a bias
        term for gradient descent.
    """
    dataframe['bias'] = 1
    return dataframe


def add_squares(dataframe, categorical_features):
    """
        For every feature in the dataframe, adds a feature that corresponds to the entrywise
        square, and names that new feature <feature>_squared
    """
    for feature in dataframe.keys():
        if feature not in categorical_features:
            dataframe[feature + "_squared"] = dataframe[feature].apply(lambda x: x * x)
    return dataframe


def add_products(dataframe, categorical_features):
    """
        For every two **distinct** features in the dataset, adds a feature that corresponds to the entrywise
        product of the two features. The new feature is named "<feature1> * <feature2>
    """

    real_features = [feature for feature in dataframe.keys() if feature not in categorical_features]

    def new_feature_naming(pair):
        return pair[0] + " * " + pair[1]

    for feature_pair in itertools.combinations(real_features, 2):  # all unordered pairs of real features, no repeats
        dataframe[new_feature_naming(feature_pair)] = dataframe[feature_pair[0]] * dataframe[feature_pair[1]]

    return dataframe


def add_products_squares(dataframe, categorical_features):
    """
        For every two **not necessarily distinct** features in the dataset, adds a feature that corresponds
        to the entrywise product of the two features (which can also be a square.
        The new feature is named "<feature1> * <feature2> if the two features are distinct and <feature>_squared
        otherwise.
    """

    real_features = [feature for feature in dataframe.keys() if feature not in categorical_features]

    def new_feature_naming(pair):
        if pair[0] == pair[1]:
            return pair[0] + "_squared"
        else:
            return pair[0] + " * " + pair[1]

    for feature_pair in itertools.combinations_with_replacement(real_features, 2):  # all unordered pairs of
        # real features, with repeats (i.e. squares are allowed)
        dataframe[new_feature_naming(feature_pair)] = dataframe[feature_pair[0]] * dataframe[feature_pair[1]]

    return dataframe


def drop_outliers(dataframe, nb_stdevs):
    """
    For the given dataframe, drops all data points where at least one of the real-valued features is at least
    nb_stdev standard deviations away from the mean. Repeats the process until no such datapoint remains (as
    the stdev and mean change with every iteration).

    :param dataframe: PANDAS dataframe to be cleaned
    :param nb_stdevs: Number of standard deviations away from mean, past which points are removed
    :return: A reference to the cleared dataframe
    """

    iteration_flag = True
    while iteration_flag:
        iteration_flag = False
        for feature in dataframe.keys():
            mean = np.mean(dataframe[feature])
            stdev = np.std(dataframe[feature])

            # finding out row indices of points to be dropped
            to_drop = dataframe[(mean - (stdev * nb_stdevs) > dataframe[feature]) |
                                (dataframe[feature] > mean + (stdev * nb_stdevs))].index

            if not to_drop.empty:
                iteration_flag = True  # if there are elements to be dropped

            # dropping all points outside of the correct range
            dataframe.drop(to_drop, inplace=True)

    return dataframe


def display_stats(dataframe, categorical_features):
    """
        Displays the main statistical properties of the dataframe. For each feature we display the mean,
        the standard deviation, and the excess kurtosis (AKA Fischer kurtosis).
    """
    for feature in dataframe.keys():
        values = dataframe[feature]
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


