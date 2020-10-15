from implementations import *
import numpy as np
from proj1_helpers import *

if __name__ == "__main__":
    


    #load data
    DATA_TRAIN_PATH = '../data/train.csv' #download train data and supply path here 
    y, x, ids = load_csv_data(DATA_TRAIN_PATH)

    #add constant term
    tx = np.c_[np.ones((y.shape[0], 1)), x]

    #print(y.shape, tx.shape, ids.shape)
    
    for i in range(tx.shape[1]):
        idx = tx[:,i] > -999
        mean = np.mean(tx[idx,i])
        tx[idx==False,i] = mean
    
    x_train, y_train, x_test, y_test = split_data(tx, y, 0.8, seed=1)
    
    gamma = 0.1
    max_iters = 200
    initial_w = np.zeros(31)

    gradients = [least_squares_GD(y_train, x_train, initial_w, max_iters,gamma),
                 least_squares_SGD(y_train, x_train, initial_w, max_iters, gamma)]
    gradients_names = ["Gradient Descent","Stochastic Gradient Descent"]
    print('-----Without standardization-----')
    for i in range (len(gradients)):
        w,loss = gradients[i]
        print("{name}, w*={w}, loss={l}\n".format(name=gradients_names[i],w=w, l=loss))