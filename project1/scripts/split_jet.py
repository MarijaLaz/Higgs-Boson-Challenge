import numpy as np
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


      
acc_LR = []
total_loss_te_LR = []
acc_RLR = []
total_loss_te_RLR = []

gamma = 1e-6
max_iters = 2000
lambda_ = 0.001
k_fold = 5
degree = 3

print("Standardizing the data...")
x_train[0], mean, std = standardize(x_train[jet_num])
print("Poly expand")
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
