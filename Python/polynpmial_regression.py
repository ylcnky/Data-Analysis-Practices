# -*- coding: utf-8 -*-

## POLYNOMIAL REGRESSION ##

# Data Preprocessing

# IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING DATASET
dataset = pd.read_csv('Position_Salaries.csv')
# Get all the rows and get all columns except last one
# First : represent rows, second : represent columns
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# No need to create training and test sets due to low amount of data

# FITTING LINEAR REGRESSION DATASET
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# FITTING THE POLYNOMIAL REGRESSION MODEL
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# VISUALIZING THE LINEAR REGRESSION RESULTS
plt.scatter(X, y)
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

# VISUALIZING THE POLYNOMIAL REGRESSION RESULTS
plt.scatter(X, y)
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'red')
plt.title('Truth or Bluff (Polynomial)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

# PREDICTING BASED ON LINEAR REGRESSION
lin_reg.predict(6.5)

# PREDICTING BASED ON POLYNOMIAL REGRESSION
lin_reg2.predict(poly_reg.fit_transform(6.5))