#Implementation functions
#all functions return (w, loss)

import numpy as np


#----------------------------------------------------------
# Loss Functions

def compute_loss_MSE(y, tx, w):
    """Calculate the loss MSE"""
    return np.sum(np.square(y-tx@w))/(y.shape[0]*2)

def compute_loss_MAE(y, tx, w):
    """Calculate the loss MAE"""
    return np.sum(abs(y-tx@w))/(y.shape[0])

#----------------------------------------------------------
# Gradients & Regressions

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx @ w
    return -1/y.shape[0] * (tx.T @ e)

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - tx @ w
    return -1/y.shape[0] * (tx.T @ e)


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y,tx,w)
        loss = compute_loss_MSE(y,tx,w)
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
    return ws[max_iters-1], losses[max_iters-1]
    

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_n,tx_n in batch_iter(y, tx, 1):
            gradient = compute_stoch_gradient(y_n, tx_n, w)
            w = w - gamma * gradient
            loss = compute_loss_MSE(y_n, tx_n, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)
    return ws[max_iters-1], losses[max_iters-1]
       
    
def least_squares(y, tx):
    """Least squares regression using normal equations"""
    w = np.linalg.solve(tx.T@tx,tx.T@y)
    loss = compute_loss_MSE(y, tx, w)
    return w,loss



def ridge_regression(y, tx, lambda_):
    #Ridge regression using normal equations
    I = np.identity(tx.shape[1])             #I(dxd)
    l = 2*lambda_*tx.shape[0]                #lambda' = 2*lambda*N
    
    A = tx.T@tx + l*I
    B = tx.T@y
    w_ridge = np.linalg.solve(A,B)
    
    #calculate loss with mse
    err = y-tx.dot(w_ridge)
    mse = np.mean(err**2)/2
    
    return w_ridge, mse



def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""

def reg_logistic_regressions(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""

    
    
#----------------------------------------------------------
#Feature processing

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    phi = np.zeros((x.shape[0], degree+1)) #degree + 1 because we want to go from 0 to degree
    for d in range(degree+1):   
         phi[:,d] = x**d
    return phi
    

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    r = int(ratio*x.shape[0])
    indices = np.random.permutation(np.arange(x.shape[0]))
    training_idx, test_idx = indices[:r], indices[r:]
   
    x_train = x[training_idx]
    y_train = y[training_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    
    return x_train, y_train, x_test, y_test

#----------------------------------------------------------
# Helpers 

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
            
def standardize(x):
    mean=np.mean(x,axis=0)
    std = np.std(x,axis=0)
    return (x-mean)/std