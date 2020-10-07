#Implementation functions
#all functions return (w, loss)

import numpy as np


def least_squares_GD(y, tx, initial w, max_iters, gamma):
    #Linear regression using gradient descent

def least_squares_SGD(y, tx, initial w, max_iters, gamma):
    #Linear regression using stochastic gradient descent

    
#********************************************************************    
    
def least_squares(y, tx):
    #Least squares regression using normal equations
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A,b)
    
    #calculate loss with mse
    e = y - tx @ w
    mse =  np.mean(e**2)/2
    
    # returns mse, and optimal weights
    return mse, w

#********************************************************************

def ridge_regression(y, tx, lambda_):
    #Ridge regression using normal equations
    I = np.identity(tx.shape[1])             #I(dxd)
    l = 2*lambda_*tx.shape[0]                #lambda' = 2*lambda*N
    
    A = tx.T@tx + l*I
    B = tx.T@y
    w_ridge = np.linalg.solve(A,B)
    
    err = y-tx.dot(w_ridge)
    mse = np.mean(err**2)/2
    
    return mse, w_ridge

#********************************************************************


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    #Logistic regression using gradient descent or SGD

def reg_logistic_regressions(y, tx, lambda_, initial_w, max_iters, gamma):
    #Regularized logistic regression using gradient descent or SGD