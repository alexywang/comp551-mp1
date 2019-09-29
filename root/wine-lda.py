from data.wine import wine_import as importer
from models import lda as model
from data import data_manipulation as manip
from validation import feature_selection as validation

BIN_FEATURE = 'quality_bin'

def train_and_predict(training, validation, binary_feature=BIN_FEATURE):
    prediction_results = model.predict(training, validation, binary_feature)
    real_results = list(validation[binary_feature])
    correct_count = 0
    for i in range(0, len(prediction_results)):
        if prediction_results[i] == real_results[i]:
            correct_count += 1

    accuracy = correct_count/len(real_results)
    return accuracy

acc_function = train_and_predict

dataset = importer.get_data('winequality-red.csv')
dataset = manip.drop_outliers(dataset, 5)
dataset = manip.normalize(dataset, BIN_FEATURE)

print('DEFAULT FEATURES:', validation.evaluate_acc(dataset, train_and_predict, BIN_FEATURE))


dataset = manip.add_squares(dataset, BIN_FEATURE)
best = validation.forward_search(dataset, BIN_FEATURE, acc_function)

print('WITH FORWARD SEARCH:', best)


