import pandas as pd
import matplotlib.pyplot as plt


def get_data(path="winequality-red.csv"):
    """
        Reads data from winequality-red.csv and applies treatment to the "quality" feature
        in order to turn it into a binary category.

        The "quality" feature is thus renamed "quality_bin"
    """
    raw_data = pd.read_csv(path, sep=";")

    treated_data = raw_data

    # Applying categorical classification to the quality of wines (assignment requirement)
    treated_data['quality_bin'] = treated_data['quality'].apply(lambda x: 1 if x >= 6 else 0)
    treated_data = treated_data.drop(columns='quality') # drops the original quality column

    # One time replicable shuffle
    treated_data = treated_data.sample(n=len(treated_data), random_state=42).reset_index(drop=True)

    return treated_data

get_data()

