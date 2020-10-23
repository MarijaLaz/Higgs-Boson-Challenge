# functions for computing loss and gradients 

import numpy as np


#----------------------------------------------------------
# Loss Functions

def compute_loss_MSE(y, tx, w):
    """Calculate the loss MSE"""
    return np.mean(np.square(y-tx.dot(w),dtype=np.float128))*2  #1. / 2 * np.mean(e ** 2)    e = y - tx.dot(w)

def compute_loss_MAE(y, tx, w):
    """Calculate the loss MAE"""
    return np.sum(abs(y-tx@w))/(y.shape[0])

def compute_loss_LG(y, tx, w):
    """Compute the loss: negative log likelihood."""
    txw = tx@w
    A = y*txw
    B = np.logaddexp(0,txw)
    #print(y)
    return np.sum(B-A)/y.shape[0]

#----------------------------------------------------------
# Gradients 

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - np.dot(tx, w)
    return -1/y.shape[0] * np.dot(tx.T, e)

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - np.dot(tx, w)
    return -1./float(y.shape[0]) * np.dot(tx.T, e)

def calculate_gradient_LR(y, tx, w):
    """Compute the gradient of loss for Logistic Regression models."""
    return tx.T@(sigmoid(tx@w)-y)

#----------------------------------------------------------
# Logistic 

def sigmoid(t):
    """Apply the sigmoid function on t."""
    #e_t =scipy.special.expit(-t)
    sig = np.exp(-np.logaddexp(0,-t))
    return sig


def calculate_hessian(y, tx, w):
    """Return the Hessian of the loss function."""
    one = np.ones(tx.shape[0]).T
    S = np.diag(np.diag(sigmoid(tx@w)*(one-sigmoid(tx@w))))
    return tx.T@S@tx



            
            





