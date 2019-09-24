import numpy as np
from data.wine import wine_import as importer
from data import data_manipulation as cleaner
from models import gradient_descent as model
from validation import feature_selection as features

BIN_FEATURE = 'quality_bin'

# perform all data manipulation and feature engineering
def process_data(dataframe, categorical_data="quality_bin"):
    step = 0
    processed = cleaner.normalize(dataframe, categorical_data)
    processed = cleaner.add_bias(processed)
    # processed = add_squares(processed, categorical_data)
    # processed = add_products(processed, categorical_data)
    # processed = add_products_squares(processed, categorical_data)
    # processed = drop_outliers(processed, 10)
    return processed


# A final wrapper function to be fed to k-fold validation. Returns the prediction accuracy on the validation set
def train_and_predict(training_data, validation_data, binary_feature=BIN_FEATURE):
    training_results = model.fit(training_data)  # (weights, fitness)
    weights = training_results[0]
    fitness = training_results[1]

    # Generate predictions for all validation data points
    prediction_vector = model.predict(validation_data, weights, binary_feature)
    prediction_results = list(np.reshape(prediction_vector, len(prediction_vector))) # TODO: Fix hacky list stuff

    # Compare predictions for real results
    real_results = list(validation_data[binary_feature])

    correct_count = 0
    for i in range(0, len(prediction_results)):
        if prediction_results[i] == real_results[i]:
            correct_count += 1

    accuracy = correct_count/len(real_results)
    return accuracy


raw_data = importer.get_data('winequality-red.csv')
dataset = process_data(raw_data, BIN_FEATURE)
print('starting validation...')

gradient = train_and_predict

best_model = features.get_best_features(dataset, BIN_FEATURE, gradient)