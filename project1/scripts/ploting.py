# code for testing range of a fiven hyperparameter and create boxplot


import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import *
from utils.preprocessing import *
from utils.crossvalidation import *



#load data
DATA_TRAIN_PATH = '../data/train.csv' #download train data and supply path here 
y, x, ids = load_csv_data(DATA_TRAIN_PATH)


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

gamma = 1e-6
max_iters = 2000
lambdas = np.logspace(-4,0,20)
k_fold = 5
degree = 3

# add offset term
x_train[0] = np.c_[np.ones((y_train[0].shape[0], 1)), x_train[0]]
# expand features
jet_0_tx_poly = build_poly(x_train[0], degree)
# initial w vector
initial_w = np.random.randn(jet_0_tx_poly.shape[1]) #np.zeros(jet_0_tx_poly.shape[1])

k_indices = build_k_indices(y_train[0], k_fold, 1)

total_loss_te = []
accuracy = []
variances = []

for lambda_ in lambdas:
	loss_te_tmp = []
	for k in range(k_fold):
		acc, loss_te_LR = cross_validation(y_train[0], jet_0_tx_poly, k_indices, k, initial_w, 'reg_logistic_regression', max_iters, gamma, lambda_)
		accuracy.append(acc)
		loss_te_tmp.append(loss_te_LR)
	total_loss_te.append(loss_te_tmp)
	variances.append(loss_te_LR)
	
	
plt.boxplot(total_loss_te)
plt.show(block=True)
 	
	



