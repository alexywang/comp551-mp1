import pandas as pd
import numpy as np

LEARNING_RATE = 0.00005
TOLERANCE = 0.0005


def _lgc(x):
    """
    The logistic function.
    :param x: Independent variable
    :return: Real-valued logistic function of x.
    """
    return 1.0 / (1.0 + np.exp(-1.0 * x))  # logistic function


def test_learning_rates_tolerances(training_data, binary_feature):
    learning_rates = [0.1, 0.05, 0.02, 0.01, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002,
                      0.0001, 0.00005, 0.00002, 0.00001]

    tolerances = [0.1, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]

    repeats = 5

    print("learning_rate, tolerance, steps")

    for tolerance in tolerances:
        for learning_rate in learning_rates:
            steps = 0

            for i in range(repeats):
                steps += _gradient_ascent(training_data, log_likelihood_gradient, learning_rate, tolerance,
                                 binary_feature=binary_feature, nb_steps_only=True)
            print(learning_rate, ",", tolerance, ",", (steps + 0.0) / repeats)



def _gradient_ascent(data, fitness_gradient, learning_rate = LEARNING_RATE,
                     tolerance = TOLERANCE, binary_feature="quality_bin", opt_print=False,
                     nb_steps_only = False, step_limit = 5000):
    """
    Performs a gradient ascent on the given function.

    :param data: The data over which the fitness gradient is evaluated.
    :param fitness_gradient: The gradient of the fitness function to be optimized. Must be a function of the data
        and of the current weights being attributed in the model.
    :param learning_rate: Gradient descent learning rate.
    :param tolerance: The minimum magnitude of change from step to step necessary to make the gradient descent stop.
    :return: A numpy vector containing the weights for all of the desired parameters after the gradient ascent.
    """
    weights = pd.DataFrame.from_dict({feature: [np.random.normal()] for feature in data.keys()})
    weights = weights.drop(binary_feature, axis='columns')
    weights_static = False

    steps = 0

    while not weights_static and steps < step_limit :
        gradient = fitness_gradient(data, weights, binary_feature)

        adjustment_norm = np.linalg.norm(gradient) * learning_rate
        if opt_print:
            print(adjustment_norm)

        if adjustment_norm < tolerance:
            weights_static = True

        else:
            weights = weights + learning_rate * gradient

        steps += 1


    if not nb_steps_only:
        return weights

    else:
        return steps


def fit(training_data, binary_feature, opt_print=False):
    """
    A wrapper for the gradient ascent function.

    :param training_data: The data to be trained over. Expected to be in PANDAS Dataframe form.
    :return: A tuple containg the weights of the model, followed by the final value of the fitness function.
    """
    weights = _gradient_ascent(training_data, log_likelihood_gradient, LEARNING_RATE, TOLERANCE,
                               binary_feature=binary_feature, opt_print=opt_print)
    fitness = log_likelihood(training_data, weights, binary_feature)
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
    Returns the gradient of the log-likelihood logistic function. The objective
    is to maximize this function through gradient ascent.

    :param binary_feature: The binary feature that we are trying to decide over.
    :param data: The data being used in order to train the model. Expected to be in PANDAS dataframe form.
    :param weights: The current estimated weights of the model.
    :return: A column vector containing the gradient of the log likelihood.
    """
    ground_result = np.array(data[binary_feature])
    predictive_info = data.drop(binary_feature, axis="columns")

    pointwise_lgc_activation = _lgc(np.dot(predictive_info, weights.transpose())).transpose()

    ll_gradient = np.dot(predictive_info.transpose(), (ground_result - pointwise_lgc_activation).transpose())
    return ll_gradient.transpose()


def predict(data, weights, binary_feature="quality_bin"):
    """
    Issues a prediction on the feature binary_feature for all of the data points contained
    in data, based on model weights.

    :param data: A PANDAS dataframe (or other array) containing data points on which a decision must be made.
    It is expected that the columns are dedicated to features, and the rows are dedicated to the data points
    to be evaluated.
    :param weights: The model weights to be used to perform the prediction.
    :return: A vector containing the predicted categorical information.
    """

    predictive_info = data.drop(binary_feature, axis="columns")
    decision_select = lambda x: 1 if x >= 0 else 0
    return np.array([decision_select(x) for x in np.dot(predictive_info, weights.transpose())]).transpose()


def predict_with_prob(data, weights, binary_feature="quality_bin"):
    """
    Returns the log-probability on the feature binary_feature for all of the data points contained
    in data, based on model weights.

    :param data: A PANDAS dataframe (or other array) containing data points on which a decision must be made.
    It is expected that the columns are dedicated to features, and the rows are dedicated to the data points
    to be evaluated.
    :param weights: The model weights to be used to perform the prediction.
    :return: A vector containing the estimated log-probability for all given datapoints.
    """

    predictive_info = data.drop(binary_feature, axis="columns")
    return np.dot(predictive_info, weights.transpose())
