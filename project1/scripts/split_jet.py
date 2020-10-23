import numpy as np
from utils.helpers import *
from utils.preprocessing import *
from utils.crossvalidation import *




#load data
DATA_TRAIN_PATH = '../data/train.csv' #download train data and supply path here 
y, x, ids = load_csv_data(DATA_TRAIN_PATH)

#add constant term
#tx = np.c_[np.ones((y.shape[0], 1)), x]

print('Shape of y => {sy} \nShape of x => {sx} \n'.format(sy=y.shape, sx=x.shape))
      
x_train, indexes = split_jet(x)
y_train = labels_jet(y, indexes)

x_train[0] = removeNaN(x_train[0])
x_train[1] = removeNaN(x_train[1])
#no nan values for jet 2 and jet 3
      
replaceNaN(x_train[0])
replaceNaN(x_train[1])
replaceNaN(x_train[2])
replaceNaN(x_train[3])
      
x_train[0] = standardize(x_train[0])
x_train[1] = standardize(x_train[1])
x_train[2] = standardize(x_train[2])
x_train[3] = standardize(x_train[3])

      
acc_LR = []
total_loss_te_LR = []
acc_RLR = []
total_loss_te_RLR = []

gamma = 1e-6
max_iters = 2000
lambda_ = 0.001
k_fold = 5
degree = 3

x_train[0] = np.c_[np.ones((y_train[0].shape[0], 1)), x_train[0]]
jet_0_tx_poly = build_poly(x_train[0], degree)
initial_w = np.random.randn(jet_0_tx_poly.shape[1]) #np.zeros(jet_0_tx_poly.shape[1])

k_indices = build_k_indices(y_train[0], k_fold, 1)

for k in range(k_fold):
    acc, loss_te_LR = cross_validation(y_train[0], jet_0_tx_poly, k_indices, k, initial_w, 'reg_logistic_regression', max_iters, gamma, lambda_)
    acc_LR.append(acc)
    total_loss_te_LR.append(loss_te_LR)

print('Parameters : gamma={g}, lambda={l}, degree={d}'.format(g=gamma,l=lambda_,d=degree))
print('Accuracy for each k_fold',acc_LR)
print('Loss for each k_fold',total_loss_te_LR)
