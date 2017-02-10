# -*- coding: utf-8 -*-

# Data Preprocessing

# IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING DATASET
dataset = pd.read_csv('Data.csv')
# Get all the rows and get all columns except last one
# First : represent rows, second : represent columns
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# SPLITTING THE DATA INTO TRAINING AND TEST SETS
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# FEATURE SCALING
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''




