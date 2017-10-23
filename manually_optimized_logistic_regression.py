import numpy as np
import os
import sys
import pdb
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import scipy.special
import scipy.optimize



def logistic_regression_log_likelihood(beta, X, y):
    # get beta into correct format
    beta_format = np.transpose(np.asmatrix(beta))
    x_beta = np.dot(X, beta_format)
    # Compute probability each data point belongs to class 1
    prob_1 = np.squeeze(np.asarray(scipy.special.expit(x_beta)))
    # Compute probability each data point belongs to class 0
    prob_0 = 1.0 - prob_1
    # Return -log likelihood
    return -np.sum(np.log(prob_1)*y + np.log(prob_0)*(1-y))

# Take the gradient of beta
def logistic_regression_gradient(beta, X, y):
    # get beta into correct format (a matrix)
    beta_format = np.transpose(np.asmatrix(beta))
    x_beta = np.dot(X, beta_format)
    # Compute probability each data point belongs to class 1
    prob_1 = np.squeeze(np.asarray(scipy.special.expit(x_beta)))

    # Initialize gradient
    gradient = np.zeros(X.shape[1])
    # Compute gradient at each element
    for index in range(len(gradient)):
        gradient[index] = np.sum(X[:,index]*y) - np.sum(X[:,index]*prob_1)
    return -gradient


def manual_logistic_regression(X,y):
    beta = np.zeros(X.shape[1])  # Initialize betas to zero vec

    # Run LBFGS
    ## This uses 'logistic_regression_gradient' and 'logistic_regression_log_likelihood functions'
    val = scipy.optimize.fmin_l_bfgs_b(logistic_regression_log_likelihood, beta,fprime=logistic_regression_gradient, args=(X,y))
    if val[2]['warnflag'] != 0:
        print('Scipy optimization did not converge successfully')
    else:
        print('Successful convergence')
        print('Negative log likelihood: ' + str(val[1]))
        print('Coefficient vector: ' + str('\t'.join(val[0].astype(str))))









#############################################################
# Create simulated data
#############################################################
n = 300  # number of samples
k = 4  # Number of features
y = np.random.randint(2, size=n)  # Randomly generated response vector
X = np.random.normal(size=(n,k))  # Randomly generated design matrix


###############################################################
#  Run logistic regression using built in package for comparison
###############################################################
X2 = sm.add_constant(X)  # Inlcude intercept
est = sm.Logit(y, X2)
est2 = est.fit()
print(est2.summary())


##################################################################
#  Do logistic regresssion by:
####### 1. Taking gradient
####### 2. Performing approximate second order optimization with LBFGS
###################################################################
manual_logistic_regression(X2,y)
