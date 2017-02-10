# -*- coding: utf-8 -*-

# Data Preprocessing

# IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING DATASET
dataset = pd.read_csv('Salary_Data.csv')
# Get all the rows and get all columns except last one
# First : represent rows, second : represent columns
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# SPLITTING THE DATA INTO TRAINING AND TEST SETS
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# FEATURE SCALING
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

# FITTING SIMPLE LINEAR REGRESSION TO THE TRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# PREDICTING TEST SET RESULTS
y_pred = regressor.predict(X_test)

# VISUALIZING THE TRAINING TEST RESULTS
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# VISUALIZING THE TEST SET RESULTS
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()