import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_proportion():
    """
        reads data from breast cancer data document. Adds data to treated_data, finds the last column which
        indicates Malignant or Benign and drops it.
    """

    raw_data = pd.read_csv("breast-cancer-wisconsin.csv")
    treated_data = raw_data
    treated_data = (treated_data.loc[treated_data["compactness"] != '?'])
    treated_data.reset_index(drop=True, inplace=True)
    b = (treated_data.loc[treated_data["BorM"] == 2])  # Get all rows that are Benign
    m = (treated_data.loc[treated_data["BorM"] == 4])  # Get all rows that a malignant
    b.reset_index(drop=True, inplace=True)  # Reset the index pos's
    m.reset_index(drop=True, inplace=True)  # Reset the index pos's
    b_rows = b['BorM'].count()  # number of rows where BorM = 2
    m_rows = m['BorM'].count()  # number of rows where BorM = 4
    treated_rows = treated_data['BorM'].count()  # total number of rows
    py_b = b_rows / treated_rows  # proportion of rows that are benign
    py_m = m_rows / treated_rows  # proportion of rows that are malignant
    print(py_b)
    print(py_m)

    # CALCULATE MEAN FOR EACH COLUMN VECTOR

    # MEANS FOR BENIGN SET
    radius_mean_b = b.radius.mean()
    texture_mean_b = b.texture.mean()
    peremiter_mean_b = b.peremiter.mean()
    area_mean_b = b.area.mean()
    smoothness_mean_b = b.smoothness.mean()
    compactness_mean_b = b.compactness.mean()
    concavity_mean_b = b.concavity.mean()
    concavepoints_mean_b = b.concavepoints.mean()
    symmetry_mean_b = b.symmetry.mean()

    # MEANS FOR MALIGNANT SET
    radius_mean_m = m.radius.mean()
    texture_mean_m = m.texture.mean()
    peremiter_mean_m = m.peremiter.mean()
    area_mean_m = m.area.mean()
    smoothness_mean_m = m.smoothness.mean()
    compactness_mean_m = m.compactness.mean()
    concavity_mean_m = m.concavity.mean()
    concavepoints_mean_m = m.concavepoints.mean()
    symmetry_mean_m = m.symmetry.mean()


print(get_proportion())


def logodds():
    print(logodds())
