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
    return accuracy


dataset = importer.get_data('breast-cancer-wisconsin.csv')
# dataset = validation.variance_threshold(dataset, BIN_FEATURE)
# datset = validation.remove_correlated(dataset, BIN_FEATURE)
dataset = manip.drop_outliers(dataset, 5)
dataset = manip.normalize(dataset, BIN_FEATURE)
# dataset = manip.add_squares(dataset, BIN_FEATURE)
dataset = manip.add_bias(dataset)
accuracy_function = train_and_predict

print('DEFAULT FEATURES: ', validation.evaluate_acc(dataset, train_and_predict, BIN_FEATURE))

best = validation.forward_search(dataset, BIN_FEATURE, accuracy_function)

print('WITH FORWARD SEARCH:', best)

# Train our best subset of features finally on the entire dataset.

best_features = ['shape_uniformity', 'single_epithelial_size', 'bland_chromatin', 'mitoses', 'class', 'clump_thickness_squared', 'size_uniformity_squared', 'shape_uniformity_squared', 'single_epithelial_size_squared', 'bare_nuclei_squared']
final_dataset = validation.filter_features(dataset, BIN_FEATURE, best_features)
acc = validation.evaluate_acc(final_dataset, accuracy_function, BIN_FEATURE)
print(acc)
print('FINAL MODEL:', model.fit(final_dataset, BIN_FEATURE))



