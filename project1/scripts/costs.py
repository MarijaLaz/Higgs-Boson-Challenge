# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae or rmse
    """
    
    # compute loss by MSE
    error = y - tx.dot(w)
    Loss = np.mean(e**2)/2

    # compute loss by MAE
    
    
    #compute loss by RMSE
    
    
    
    return Loss
    
 