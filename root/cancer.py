import numpy as np
from data.cancer import bc_import as importer
from data import data_manipulation as manip
from models import gradient_descent as model
from validation import feature_selection as validation


raw_data = importer.get_data()
print(raw_data)