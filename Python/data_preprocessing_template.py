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
                
# TAKING CARE OF MISSING DATA
# To fill the missing data, we fill the missing part with the mean of other values

# Import the Imputer class for missing data
from sklearn.preprocessing import Imputer
# Whichever includes "NaN" apply the strategy mean for columns (0)
imputer = Imputer(missing_values="NaN", strategy='mean', axis = 0)
# Dont select the entire matrix, select only the independent columns which includes "NaN" only
# If we select all columns, than the values may change
imputer = imputer.fit(X[:, 1:3])
# Put the mean value for the missing cells
X[:, 1:3] = imputer.transform(X[:, 1:3])

# ENCODING THE CATEGORICAL DATA
# This process is for converting the textual values to numerical values. Because ML needs only numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#LabelEncoder does not give good result because it return 2,3, or other number which are not good mathematical functions
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# To prevent above problem, we use OneHotEncoder to create only 0 and 1
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Last column label encoder is fine because there are only Yes and No
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# SPLITTING THE DATA INTO TRAINING AND TEST SETS
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)




