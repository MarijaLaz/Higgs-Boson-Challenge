#Implementation functions
#all functions return (w, loss)

import numpy as np


<<<<<<< HEAD
#----------------------------------------------------------
# Loss Functions

def compute_loss_MSE(y, tx, w):
    """Calculate the loss MSE"""
    return np.sum(np.square(y-tx@w))/(y.shape[0]*2)

def compute_loss_MAE(y, tx, w):
    """Calculate the loss MAE"""
    return np.sum(abs(y-tx@w))/(y.shape[0])

#----------------------------------------------------------
# Gradients 

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
=======
def least_squares_GD(y, tx, initial w, max_iters, gamma):
    #Linear regression using gradient descent

def least_squares_SGD(y, tx, initial w, max_iters, gamma):
    #Linear regression using stochastic gradient descent
>>>>>>> 30da2a019eebcc55809fb5263d127ec61a002ebd

    
#********************************************************************    
    
def least_squares(y, tx):
<<<<<<< HEAD
    """Least squares regression using normal equations"""
    w = np.linalg.solve(tx.T@tx,tx.T@y)
    loss = compute_loss_MSE(y, tx, w)
    return w,loss

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    w = np.linalg.solve(tx.T@tx+lambda_*np.identity(tx.shape[1]),tx.T@y)
    loss = compute_loss_MSE(y, tx, w) 
    return w, loss
=======
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

>>>>>>> 30da2a019eebcc55809fb5263d127ec61a002ebd

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""

def reg_logistic_regressions(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    
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