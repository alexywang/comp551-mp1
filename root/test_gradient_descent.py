import data.wine.wine_import as wi
import data.data_manipulation as dm

import models.gradient_descent as gd
import numpy as np

df = wi.get_data()
dm.drop_outliers(df, 4)
dm.normalize(df, "quality_bin")
# dm.add_products_squares(df, "quality_bin")
dm.add_bias(df)

gd.test_learning_rates_tolerances(df, "quality_bin")
