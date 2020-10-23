# Useful starting lines

import numpy as np
from utils.implementations import *
from utils.helpers import *
from utils.preprocessing import *
from utils.crossvalidation import *


#load data
DATA_TRAIN_PATH = '../data/train.csv' #download train data and supply path here 
y, x, ids = load_csv_data(DATA_TRAIN_PATH)

#add constant term
#tx = np.c_[np.ones((y.shape[0], 1)), x]

print('Shape of y => {sy} \n Shape of x => {sx} \n'.format(sy=y.shape, sx=x.shape))
      
         
jet_0_tx, jet_1_tx, jet_2_tx, jet_3_tx, jet_0_y, jet_1_y, jet_2_y, jet_3_y, index_0, index_1, index_2, index_3 = split_jet(y, x)

#jet_0_tx = removeNaN(jet_0_tx)
#jet_1_tx = removeNaN(jet_1_tx)
#no nan values for jet 2 and jet 3
      

replaceNaN(jet_3_tx)
jet_3_tx = standardize(jet_3_tx)

#without feature extension
acc_LR = []
total_loss_te_LR = []
acc_RLR = []
total_loss_te_RLR = []

gamma = 1e-6
max_iters = 2000
lambda_ = 0.001
k_fold = 5

initial_w = np.zeros(jet_3_tx.shape[1])

k_indices = build_k_indices(jet_3_y, k_fold, 1)


for k in range(k_fold):
    acc, loss_te_LR = cross_validation(jet_3_y, jet_3_tx, k_indices, k, initial_w, 'logistic_regression', max_iters, gamma, lambda_)
    acc_LR.append(acc)
    total_loss_te_LR.append(loss_te_LR)

print(acc_LR)
print(total_loss_te_LR)


      #with feature extension
acc_LR = []
total_loss_te_LR = []
acc_RLR = []
total_loss_te_RLR = []

gamma = 1e-6
max_iters = 2000
lambda_ = 0.001
k_fold = 5
degree=3

jet_3_tx_poly = build_poly(jet_3_tx, degree)
initial_w = np.zeros(jet_3_tx_poly.shape[1])

k_indices = build_k_indices(jet_3_y, k_fold, 1)


for k in range(k_fold):
    acc, loss_te_LR = cross_validation(jet_3_y, jet_3_tx_poly, k_indices, k, initial_w, 'logistic_regression', max_iters, gamma, lambda_)
    acc_LR.append(acc)
    total_loss_te_LR.append(loss_te_LR)

print(acc_LR)
print(total_loss_te_LR)



