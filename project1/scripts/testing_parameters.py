# code for testing range of a fiven hyperparameter and create boxplot


import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import *
from utils.preprocessing import *
from utils.crossvalidation import *


# files need to be unziped before load
DATA_TRAIN_PATH = '../data/train.csv' 


print("Load the data from csv files...")
y_train, x_train, ids_train = load_csv_data(DATA_TRAIN_PATH)


print('TRAIN : Shape of y => {sy} \n\tShape of x => {sx}'.format(sy=y_train.shape, sx=x_train.shape))

 
# splitting the data according to the jet number(0,1,2,3) and then according to the feature DER_mass_MMC
print("Splitting the train data...")
x_train, indexes_train = split_jet(x_train)
x_train, indexes_train = split_mass(x_train, indexes_train)
y_train = labels_jet(y_train, indexes_train)


print("Preprocessing the data...")
x_train = removeNaN(x_train)
x_train = addingFeatures(x_train)


print("Standardizing the data...")
x_train[6], mean, std = standardize(x_train[6])


gammas = [1e-2,1e-3,1e-5,1e-6,1e-8]
max_iters = 1000
lambdas = np.logspace(-4,0,20)
k_fold = 5
degrees = [2,3,4, 5,6,7]

k_indices = build_k_indices(y_train[6], k_fold, 1)

variances = []
best_degrees_l =[]
best_degrees_a =[]
best_losses = []
best_accuracy = []


print(x_train[0].shape)
print(x_train[1].shape)
print(x_train[2].shape)
print(x_train[3].shape)
print(x_train[4].shape)
print(x_train[5].shape)
print(x_train[6].shape)
print(x_train[7].shape)

# crossvalidation with k_fold
for gamma in gammas:
	total_loss_te = []
	total_accuracy = []
	for degree in degrees:
		loss_te_tmp = []
		accuracy = []
		for k in range(k_fold):
			# expand features
			jet_0_tx_poly = build_poly(x_train[6], degree)
			# initial w vector
			initial_w = np.random.randn(jet_0_tx_poly.shape[1])
			
			acc, loss_te_LR = cross_validation(y_train[6], jet_0_tx_poly, k_indices, k, initial_w, 'logistic_regression', max_iters, gamma)
			
			accuracy.append(acc)
			loss_te_tmp.append(loss_te_LR)
			
		total_loss_te.append(np.mean(loss_te_tmp))
		total_accuracy.append(np.mean(accuracy))
		variances.append(loss_te_LR)
		
	ind_degree_opt_l = np.argmin(total_loss_te)
	ind_degree_opt_a = np.argmax(total_accuracy)
	
	best_degrees_l.append(degrees[ind_degree_opt_l])
	best_degrees_a.append(degrees[ind_degree_opt_a])
	
	best_losses.append(total_loss_te[ind_degree_opt_l])
	best_accuracy.append(total_accuracy[ind_degree_opt_a])

print("6")
print("LOSS:")
print("Tested gammas", gammas)	
print("Best gamma", gammas[np.argmin(best_losses)])
print("Tested degrees", degrees)
print("Best degrees ", best_degrees_l)
print("-------------------------------------")
print("ACCURACY:")
print("Tested gammas", gammas)	
print("Best gamma", gammas[np.argmax(best_accuracy)])
print("Tested degrees", degrees)
print("Best degrees ", best_degrees_a)
 	
	



