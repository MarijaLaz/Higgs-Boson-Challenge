# functions for preprocessing the data

import numpy as np


def standardize(x):
    '''Standardize the data with zero mean and unit variance'''
    mean=np.mean(x,axis=0)
    std = np.std(x,axis=0)
    return (x-mean)/std

def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.ones((x.shape[0], 1)) #degree + 1 because we want to go from 0 to degree
    for d in range(degree+1):   
         phi = np.c_[phi, x**d]
    return phi
    

def split_data(x, y, ratio, seed=1):
    """Split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    r = int(ratio*x.shape[0])
    indices = np.random.permutation(np.arange(x.shape[0]))
    training_idx, test_idx = indices[:r], indices[r:]
   
    x_train = x[training_idx]
    y_train = y[training_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    
    return x_train, y_train, x_test, y_test


def split_jet(dataset):
    '''
  	dataset -> dataset that needs to be splitted
  
        Splits data into data sets according to the number of jets
        -------
        Returns
        list of 4 data sets for each jet_num
        list of 4 indexes arrays for each jet_num
    '''
    #examples
    jet_0 = []
    jet_1 = []
    jet_2 = []
    jet_3 = []
    
    #indexes
    index_0 = []
    index_1 = []
    index_2 = []
    index_3 = []

    for i in range(dataset.shape[0]):
        if dataset[i,22] == 0:
            jet_0.append(dataset[i])
            index_0.append(i)
        if dataset[i,22] == 1:
            jet_1.append(dataset[i])
            index_1.append(i)
        if dataset[i,22] == 2:
            jet_2.append(dataset[i])
            index_2.append(i)
        if dataset[i,22] == 3:
            jet_3.append(dataset[i])
            index_3.append(i)
            
    
    #removing the column jet_num which has index 22    
    jet_0 = np.delete(jet_0, 22, axis=1)
    jet_1 = np.delete(jet_1, 22, axis=1)
    jet_2 = np.delete(jet_2, 22, axis=1)
    jet_3 = np.delete(jet_3, 22, axis=1)
    
    #removing the PRI_jet_all_pt from jet_0
    jet_0 = np.delete(jet_0, -1, axis=1)
            
    return [np.array(jet_0), np.array(jet_1), np.array(jet_2), np.array(jet_3)], \
           [np.array(index_0), np.array(index_1),np.array(index_2),np.array(index_3)]

def labels_jet(y, indexes):
	'''
		y -> labels dataset
		indexes -> list of indexes for the corresponding jet datasets
		
		Splits the labels dataset according to the number of jets
		-------
		Returns
		list of 4 data sets for each jet_num
	'''
	labels = []
	for index in indexes:
		labels.append(y[index])
	return labels

def removeNaN(jet_x):
    '''Removes the columns that have nan values for each row'''
    return jet_x[:, np.any((jet_x != -999), axis=0)]


def replaceNaN(jet_x):
    '''Replaces the nan values'''
    for i in range(jet_x.shape[1]):
        idx = jet_x[:,i] > -999
        mean = np.mean(jet_x[idx,i])
        jet_x[idx==False,i] = mean
