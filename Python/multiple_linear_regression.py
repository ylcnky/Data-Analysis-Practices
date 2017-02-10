# -*- coding: utf-8 -*-

# MULTIPLE LINEAR REGRESSION
 
# IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING DATASET
dataset = pd.read_csv('50_Startups.csv')

# Get all the rows and get all columns except last one
# First : represent rows, second : represent columns

# All independent variables
X = dataset.iloc[:,:-1].values
# Dependent variable vector
y = dataset.iloc[:,4].values

# ENCODING THE CATEGORICAL DATA
# This process is for converting the textual values to numerical values. Because ML needs only numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#LabelEncoder does not give good result because it return 2,3, or other number which are not good mathematical functions
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
# To prevent above problem, we use OneHotEncoder to create only 0 and 1
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
# I dont take the column index 0
X = X[:, 1:]
         

# SPLITTING THE DATA INTO TRAINING AND TEST SETS
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# FITTING MULTIPLE LINEAR REGRESSION  TO THE TRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# PREDICTING THE TEST SET RESULTS
y_pred = regressor.predict(X_test)

# BUILDING THE OPTIMAL MODEL USING BACKWARD ELIMINATION
import statsmodels.formula.api as sm
# Include the values 1 to the beginning of X table for Backward Elimination
X = np.append(arr = np.ones((50, 1)).astype(int), values = X ,axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()





