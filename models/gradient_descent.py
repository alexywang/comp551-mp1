import pandas as pd
import numpy as np

LEARNING_RATE = 0.0001
TOLERANCE = 0.0001


def _lgc(x): return 1.0 / (1.0 + np.exp(-1.0 * x))  # logistic function


def _gradient_ascent(data, fitness_gradient, learning_rate, tolerance, binary_feature="quality_bin", opt_print=False):
    """

    :param data:
    :param fitness_gradient:
    :param learning_rate:
    :return:
    """
    weights = pd.DataFrame.from_dict({feature: [np.random.normal()] for feature in data.keys()})
    weights = weights.drop(binary_feature, axis='columns')
    weights_static = False

    while not weights_static:
        gradient = fitness_gradient(data, weights)

        adjustment_norm = np.linalg.norm(gradient) * learning_rate
        if opt_print:
            print(adjustment_norm)

        if adjustment_norm < tolerance:
            weights_static = True

        else:
            weights = weights + learning_rate * gradient

    return weights


def fit(training_data, opt_print = False):
    """

    :param training_data:
    :return:
    """
    weights = _gradient_ascent(training_data, log_likelihood_gradient, LEARNING_RATE, TOLERANCE, opt_print=opt_print)
    fitness = log_likelihood(training_data, weights)
    return weights, fitness


def log_likelihood(data, weights, binary_feature="quality_bin"):
    """

    :param binary_feature:
    :param data:
    :param weights:
    :return:
    """
    ground_result = data[binary_feature]
    predictive_info = data.drop(binary_feature, axis="columns")

    pointwise_lgc_activation = _lgc(predictive_info.dot(weights.transpose()))

    pointwise_likelihood = ground_result * np.log(pointwise_lgc_activation) + \
                           (1 - ground_result) * np.log(1 - pointwise_lgc_activation)

    pointwise_likelihood = pointwise_likelihood[0]
    return np.sum(pointwise_likelihood)


def log_likelihood_gradient(data, weights, binary_feature="quality_bin"):
    """

    :param feature:
    :param data:
    :param weights:
    :return:
    """
    ground_result = np.array(data[binary_feature])
    predictive_info = data.drop(binary_feature, axis="columns")

    pointwise_lgc_activation = _lgc(np.dot(predictive_info, weights.transpose())).transpose()

    ll_gradient = np.dot(predictive_info.transpose(), (ground_result - pointwise_lgc_activation).transpose())
    return ll_gradient.transpose()


def predict(data, weights, binary_feature="quality_bin"):
    """

    :param data:
    :param weights:
    :return:
    """

    # TODO: Fix this function
    predictive_info = data.drop(binary_feature, axis="columns")
    decision_select = lambda x: 1 if x >= 0 else 0
    return decision_select(np.dot(predictive_info, weights.transpose()))