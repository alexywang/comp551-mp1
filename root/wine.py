import numpy as np
import pandas as pd
import matplotlib as plot
import jupyter as jp
from data.wine import wine_import as importer
from data import data_manipulation as manip
from models import gradient_descent as model
from validation import feature_selection as validation
import math

BIN_FEATURE = 'quality_bin'


# perform all data manipulation and feature engineering
def process_data(dataframe, categorical_data="quality_bin", normalize=True):
    # adjust total sulfur dioxide to remaining sulfur dioxide to be fully independent
      # for i in dataframe.index:
    #     dataframe.at[i, 'total sulfur dioxide'] -= dataframe.at[i, 'free sulfur dioxide']
    # dataframe.rename(columns={'total sulfur dioxide': 'other sulfur dioxide'}, inplace=True)

    dataframe = manip.drop_outliers(dataframe, 5)

    # Taking the log of total sulfur dioxide due to massive variance
    dataframe['log total sulfur dioxide'] = dataframe['total sulfur dioxide'].apply(lambda x: math.log(x, 10))
    # Squaring fixed acidity due to non linear relationship
    dataframe['log residual sugar'] = dataframe['residual sugar'].apply(lambda x: math.log(x, 10))
    dataframe['log sulphates'] = dataframe['sulphates'].apply(lambda x: math.log(x,10))
    dataframe['log chlorides'] = dataframe['chlorides'].apply(lambda x: math.log(x,10))
    dataframe['log free sulfur dioxide'] = dataframe['free sulfur dioxide'].apply(lambda x: math.log(x, 10))

    dataframe = validation.variance_threshold(dataframe, BIN_FEATURE)
    if normalize:
        processed = manip.normalize(dataframe, categorical_data)
    dataframe = manip.add_bias(dataframe)

    return dataframe


# A final wrapper function to be fed to k-fold validation. Returns the prediction accuracy on the validation set
def train_and_predict(training_data, validation_data, binary_feature=BIN_FEATURE):
    training_results = model.fit(training_data, binary_feature)  # (weights, fitness)
    weights = training_results[0]
    fitness = training_results[1]

    # Generate predictions for all validation data points
    prediction_vector = model.predict(validation_data, weights, binary_feature)
    prediction_results = list(np.reshape(prediction_vector, len(prediction_vector)))

    # Compare predictions for real results
    real_results = list(validation_data[binary_feature])

    correct_count = 0
    for i in range(0, len(prediction_results)):
        if prediction_results[i] == real_results[i]:
            correct_count += 1

    accuracy = correct_count/len(real_results)
    return accuracy


raw_data = importer.get_data('winequality-red.csv')

dataset = process_data(raw_data, BIN_FEATURE, True)

accuracy_function = train_and_predict

print('DEFAULT FEATURES: ', validation.evaluate_acc(dataset, train_and_predict, BIN_FEATURE))

best = validation.forward_search(dataset, BIN_FEATURE, accuracy_function)
print('WITH FORWARD SEARCH:', best)

# Train our best subset of features finally on the entire dataset.
best_features = best[1]

# best_features = ['bias', 'alcohol', 'volatile acidity', 'other sulfur dioxide', 'residual sugar', 'sulphates', 'free sulfur dioxide', 'pH', 'citric acid', 'fixed acidity']
# best_features.append('log other sulfur dioxide')
# # best_features.append('log sulphates')
# # best_features.append('log residual sugar')
# best_features.remove('other sulfur dioxide')
# # best_features.remove('sulphates')
# # best_features.remove('residual sugar')

final_dataset = validation.filter_features(dataset, BIN_FEATURE, best_features)
final_dataset = final_dataset.drop('alcohol', axis='columns')

print(final_dataset)

print(final_dataset.columns)

print('Expansion:', validation.evaluate_acc(final_dataset, train_and_predict, BIN_FEATURE))
print('FINAL MODEL:', model.fit(final_dataset, BIN_FEATURE))
