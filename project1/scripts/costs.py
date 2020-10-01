# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # MSE
    #X = [1, tx]
    N = y.shape[0]
    error = y - tx.dot(w)
    Loss = (1/(2.0*N))*np.inner(error, error)

    # TODO: compute loss by MAE
    # ***************************************************
    return Loss
    
    raise NotImplementedError
    
 