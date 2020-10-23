# code for getting the final predictions for the test data

from utils.helpers import *
from utils.preprocessing import *
from utils.crossvalidation import *


print("Load the data from csv files...")

# files need to be unziped before load
DATA_TRAIN_PATH = '../data/train.csv' 
DATA_TEST_PATH  = '../data/test.csv' 
OUTPUT_PATH = '../data/results.csv' 

y_train, x_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
y_test, x_test, ids_test = load_csv_data(DATA_TEST_PATH)

print('TRAIN : Shape of y => {sy} \nShape of x => {sx}'.format(sy=y_train.shape, sx=x_train.shape))
print('TEST  : Shape of y => {sy} \nShape of x => {sx}'.format(sy=y_test.shape, sx=x_test.shape)) 
  
  
# splitting the data according to the jet number(0,1,2,3)
print("Splitting the train data...")
x_train, indexes_train = split_jet(x_train)
y_train = labels_jet(y_train, indexes_train)

print("Splitting the test data...")
x_test, indexes_test = split_jet(x_test)
y_test = labels_jet(y_test, indexes_test)

print("Preprocessing the data...")
x_train[0] = removeNaN(x_train[0])
x_train[1] = removeNaN(x_train[1])

x_test[0] = removeNaN(x_test[0])
x_test[1] = removeNaN(x_test[1])
# no nan values for jet 2 and jet 3


MAX_ITER = 2000
GAMMA = 1e-6
lambdas = [0.001, 0.001, 0.001, 0.001]
degrees = [3,3,3,3]


for jet_num in range (4):
	# replacing the nan values in columns where they are <100% present
	replaceNaN(x_train[jet_num])
	
	# standardizing the data
	x_train[jet_num] = standardize(x_train[jet_num])
	x_test[jet_num]  = standardize(x_test[jet_num])
	
	# adding the offset term
	tx_train[jet_num] = np.c_[np.ones((y_train[jet_num].shape[0], 1)), x_train[jet_num]]
	tx_test[jet_num]  = np.c_[np.ones((y_test[jet_num].shape[0], 1)), x_test[jet_num]]
	
	# feature expansion
	tx_train[jet_num] = build_poly(tx_train[jet_num], degrees[jet_num])
	tx_test[jet_num]  = build_poly(tx_test[jet_num], degrees[jet_num])
	
	# initial w vector
	initial_w = np.random.randn(tx_train[jet_num].shape[1])
	
	# training the model
	w, loss = reg_logistic_regression(y_train[jet_num], tx_train[jet_num], lambdas[jet_num], initial_w, MAX_ITER, GAMMA)
	
	# applying the w vector to the test data
	test_results = predict_labels(w, tx_test[jet_num], True)
	final[indexes_test[jet_num]] = test_results
	
	
# creating the sumbission file
print("Creating the csv file for submission...")
create_csv_submission(ids_test, final, OUTPUT_PATH)

print("DONE!")
