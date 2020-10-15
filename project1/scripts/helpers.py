import numpy as np
import scipy

#----------------------------------------------------------
# Loss Functions

def compute_loss_MSE(y, tx, w):
    """Calculate the loss MSE"""
    return np.mean(np.square(y-tx.dot(w),dtype=np.float128))*2  #1. / 2 * np.mean(e ** 2)    e = y - tx.dot(w)

def compute_loss_MAE(y, tx, w):
    """Calculate the loss MAE"""
    return np.sum(abs(y-tx@w))/(y.shape[0])

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    sig = sigmoid(np.dot(tx,w))
    A = y.T.dot(np.log(sig))
    B = (1-y).T.dot(np.log(1-sig))
    return -(A+B)

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

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T@(sigmoid(tx@w)-y)


def sigmoid(t):
    """apply the sigmoid function on t."""
    e_t =scipy.special.expit(-t)
    return 1./(1.+e_t)


def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    one = np.ones(tx.shape[0]).T
    S = np.diag(np.diag(sigmoid(tx@w)*(one-sigmoid(tx@w))))
    return tx.T@S@tx

#----------------------------------------------------------
#Feature processing


def standardize(x):
    mean=np.mean(x,axis=0)
    std = np.std(x,axis=0)
    return (x-mean)/std

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    phi = np.ones((x.shape[0], 1)) #degree + 1 because we want to go from 0 to degree
    for d in range(degree+1):   
         phi = np.c_[phi, x**d]
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
# Cross Validation

def cross_validation(y, x, k_indices, k, initial_w, lambda_, max_iters, gamma, degree):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    test_idx = k_indices[k]
    idx_tr = np.arange(len(k_indices))
    train_idx = k_indices[idx_tr != k]
    train_idx = train_idx.flatten()

    # form data with polynomial degree
    x_tr = build_poly(x[train_idx], degree)
    x_te = build_poly(x[test_idx], degree)
    y_tr = y[train_idx]
    y_te = y[test_idx]

    # ridge regression
    initial_w = np.zeros(x_tr.shape[1])
    w_r, mse = reg_logistic_regressions(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)

    # calculate the loss for train and test data
    loss_tr = compute_loss_MSE(y_tr, x_tr, w_r)
    loss_te = compute_loss_MSE(y_te, x_te, w_r)

    return loss_tr, loss_te

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


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
            
            





