import numpy as np
from data.cancer import bc_import as importer
from data import data_manipulation as manip
from models import gradient_descent as model
from validation import feature_selection as validation

BIN_FEATURE = 'class'


# A final wrapper function to be fed to k-fold validation. Returns the prediction accuracy on the validation set
def train_and_predict(training_data, validation_data, binary_feature=BIN_FEATURE):
    training_results = model.fit(training_data, BIN_FEATURE, False)  # (weights, fitness)
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
    return weights, fitness, accuracy


dataset = manip.add_bias(importer.get_data())

cov = dataset.cov().values
print(cov)
# accuracy_function = train_and_predict

# best = validation.forward_search(dataset, BIN_FEATURE, accuracy_function)
# print(best)

dataset.drop(BIN_FEATURE, axis='columns')


