
########################################################
###													 ###
###  Lending Club - Regression Analysis				 ###
###  Comparison of OLS, Lasso, and Ridge Regressions ###
###													 ###
########################################################

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


##########################
# Load and clean dataset #
##########################

data = pd.read_csv('data/LoanStats.csv')

depVar = 'int_rate'		# define which var will be the dep var
y = data[depVar]		# create y array
del data[depVar]		# delete dep var from DataFrame

y = y.values			# convert dep var to array
x = data.values			# convert indep vars to matrix


###################
# Fit Regressions #
###################

# run OLS regression
ols = linear_model.LinearRegression()
ols.fit(x, y)

# run Lasso regression
lasso = linear_model.Lasso(alpha = 1, max_iter=10000)
lasso.fit(x, y)

# run Ridge regression
ridge = linear_model.Ridge(alpha = 1, max_iter=10000)
ridge.fit(x, y)


#####################
# Plot Coefficients #
#####################

# create array for x axis 
b = range(1,len(x[0])+1)

# plot data
plt.plot(b, ols.coef_, 'r--o', label='OLS') 
plt.plot(b, lasso.coef_, 'b--', label='Lasso')
plt.plot(b, ridge.coef_, 'g--', label='Ridge')
plt.ylabel('Coefficient Value')
plt.xlabel('Coefficient Number')
plt.title('Regression Comparison')
plt.show()



	