import numpy as np
import pandas as pd



def load_split_data(ratio, seed=1):
    """Load dataset from csv files and Split the dataset based on the split ratio."""
    
    #load data
    pred_data = pd.read_csv("Data/true_ees.csv")
    y = pred_data.to_numpy()[:,1]

    data = pd.read_csv("Data/u2.csv")
    X = data.to_numpy()
    
    # split data
    np.random.seed(seed)
    r = int(ratio*X.shape[0])
    indices = np.random.permutation(np.arange(X.shape[0]))
    training_idx, test_idx = indices[:r], indices[r:]
   
    x_train = X[training_idx]
    y_train = y[training_idx]
    x_test = X[test_idx]
    y_test = y[test_idx]
    
    return x_train, y_train, x_test, y_test