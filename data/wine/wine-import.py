import pandas as pd

def get_data():
    raw_data = pd.read_csv("winequality-red.csv", sep=";")
    treated_data = raw_data

    # Applying categorical classification to the quality of wines (assignment requirement)
    treated_data['quality_bin'] = treated_data['quality'].apply(lambda x: 1 if x >= 6 else 0)

    treated_data = treated_data.drop(columns='quality') # drops the original quality column
    return treated_data