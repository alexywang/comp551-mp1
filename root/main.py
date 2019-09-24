from data.wine import wine_import as importer
from data import data_manipulation as cleaner
from models import gradient_descent as model
from validation import k_fold

raw_data = importer.get_data('winequality-red.csv')
dataset = cleaner.process_data(raw_data)
print('starting validation...')

gradient = model.train_and_predict

avg_acc = k_fold.kfold_validate(dataset, 5, gradient, 'quality_bin')

print(avg_acc)

