# mandatory implementation functions

import numpy as np
from utils.loss_gradient import *
from utils.helpers import batch_iter


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y,tx,w)
        loss = compute_loss_MSE(y,tx,w)
        w = w - gamma * gradient
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
            ws.append(w)
            losses.append(loss)
    return ws[max_iters-1], losses[max_iters-1]
       
    
def least_squares(y, tx):
    """Least squares regression using normal equations"""
    w = np.linalg.solve(tx.T@tx,tx.T@y)
    loss = compute_loss_MSE(y, tx, w)
    return w,loss



def ridge_regression(y, tx, lambda_):
    '''Ridge regression using normal equations'''
    I = np.identity(tx.shape[1])             #I(dxd)
    l = 2*lambda_*tx.shape[0]                #lambda' = 2*lambda*N
    
    A = tx.T@tx + l*I
    B = tx.T@y
    w_ridge = np.linalg.solve(A,B)
    err = y-tx.dot(w_ridge)
    mse = compute_loss_MSE(y, tx, w_ridge)
    return w_ridge, mse



def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent"""
    ws = [initial_w]
    losses = []
    w = initial_w
    # changing labels values -1 -> 0, 1 -> 1
    y_ = (1+y)/2
    for n_iter in range(max_iters):
        gradient = calculate_gradient_LR(y_, tx, w)
        w = w - gamma * gradient
        loss =  compute_loss_LG(y_, tx, w)
        ws.append(w)
        losses.append(loss)
    return ws[-1],losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent"""
    ws = [initial_w]
    losses = []
    w = initial_w
    # changing labels values -1 -> 0, 1 -> 1
    y_ = (1+y)/2
    for n_iter in range(max_iters):
        gradient = calculate_gradient_LR(y_, tx, w) + 2*lambda_*w
        w = w - gamma * gradient
        loss =  compute_loss_LG(y_, tx, w)+ lambda_*np.linalg.norm(w)
        ws.append(w)
        losses.append(loss)
        # convergence criteria
        if n_iter > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            break
    return ws[-1],losses[-1]
    
