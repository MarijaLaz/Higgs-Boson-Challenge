# functions for preprocessing the data

import numpy as np
from utils.features import *

def standardize(x,mean=None,std=None,test=False):
    '''
    	test -> False : used for standardizing the train data
    		True  : used for standardizing test data
    	Standardize the data with zero mean and unit variance
    '''
    if not test:
        mean=np.mean(x,axis=0)
        std = np.std(x,axis=0)
    return (x-mean)/std, mean, std

def build_poly(x, degree):
    """Polynomial basis function for input data x, for j=0 up to j=degree."""
    phi = np.ones((x.shape[0], 1)) 
    #degree + 1 because we want to go from 0 to degree
    for d in range(1,degree+1):   
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
        List of 4 data sets for each jet_num
        List of 4 indexes arrays for each jet_num
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

def split_mass(jets,indexes):
    '''
        jets    -> the spplitted data sets according to jets
        indexes -> the indexes for the coresponding jet data sets
        
        Splits the data sets according to the value of the column DER_mass_MMC(nan and !nan)
        ------
        Returns
        List of 8 data sets
        List of arrays of indexes that correspond to the starting data set
    '''
    splits = []
    splitsIndex = []
    for jetNum in range (len(jets)):
        mass = jets[jetNum][:,0]
        massNan = (mass == -999)
        
        indexMassNan = indexes[jetNum][massNan]
        indexMass = indexes[jetNum][np.logical_not(massNan)]
        
        jetMassNan = jets[jetNum][massNan]
        jetMass = jets[jetNum][np.logical_not(massNan)]
        
        splits.append(np.array(jetMassNan))
        splits.append(np.array(jetMass))
        
        splitsIndex.append(np.array(indexMassNan))
        splitsIndex.append(np.array(indexMass))
    return splits, splitsIndex

def labels_jet(y, indexes):
    '''
        y       -> labels dataset
        indexes -> list of indexes for the corresponding jet datasets
        
        Splits the labels dataset according to the number of jets
        -------
        Returns
        labels -> list of data sets according to indexes
    '''
    labels = []
    for index in indexes:
        labels.append(y[index])
    return labels

def removeNaN(jets):
    '''Removes the columns that have nan values for each row'''
    result = []
    for jet in jets:
        result.append(jet[:, np.any((jet != -999), axis=0)])
    return  result


def replaceNaN(jet_x):
    '''Replaces the nan values'''
    for i in range(jet_x.shape[1]):
        idx = jet_x[:,i] > -999
        mean = np.mean(jet_x[idx,i])
        jet_x[idx==False,i] = mean

def dictionaryIndex(n):
    '''
        n -> index of the data set
        
        Function used when using the dictionary for the 8 data sets.
        -------
        Returns
        Data set 0 and 1 => index 0 in dictionary
        Data set 2 and 3 => index 1 in dictionary
        Data set 3 and 4 => index 2 in dictionary
        Data set 5 and 6 => index 3 in dictionary
    '''
    if n==0 or n==1:
        return 0
    if n==2 or n==3:
        return 1
    if n==4 or n==5:
        return 2
    if n==6 or n==7:
        return 3
        
def addingFeatures(jets):
	'''
		jets -> the 8 data sets
		Function that transforms the phi and eta features
			-> subtracts the feature PRI_tau_phi from the other phi features and then removes it
			-> the phi features values are adjusted to be between [-pi,pi]
			-> replaces with their abs value the values of the eta features if PRI_tau_eta has a negative value
		-------
		Returns
		results -> the 8 data sets after the feature transformation
	'''
    results=[]
    for i,x in enumerate(jets):
        dic = jet_feautures[i]
        # work on angels
        to_remove = x[:,dic["PRI_tau_phi"]]
        PRI_lep_phi = x[:,dic["PRI_lep_phi"]]
        PRI_met_phi = x[:,dic["PRI_met_phi"]]
        if i > 1:
            PRI_jet_leading_phi = x[:,dic["PRI_jet_leading_phi"]]
        if i > 3:
            PRI_jet_subleading_phi = x[:,dic["PRI_jet_subleading_phi"]]

        # rotating the angles
        PRI_lep_phi = np.abs(PRI_lep_phi - to_remove)
        PRI_met_phi = np.abs(PRI_met_phi - to_remove)
        if i > 1:
            PRI_jet_leading_phi = np.abs(PRI_jet_leading_phi - to_remove)
        if i > 3:
            PRI_jet_subleading_phi = np.abs(PRI_jet_subleading_phi - to_remove)

        # adjusting the angles to have values in the interval [-pi,pi]
        PRI_lep_phi[PRI_lep_phi>np.pi] = 2*np.pi-PRI_lep_phi[PRI_lep_phi>np.pi]
        PRI_met_phi[PRI_met_phi > np.pi] = 2*np.pi-PRI_met_phi[PRI_met_phi > np.pi]
        if i > 1:
            PRI_jet_leading_phi[PRI_jet_leading_phi > np.pi] = 2*np.pi-PRI_jet_leading_phi[PRI_jet_leading_phi > np.pi]
        if i > 3:
            PRI_jet_subleading_phi[PRI_jet_subleading_phi > np.pi] = 2*np.pi-PRI_jet_subleading_phi[PRI_jet_subleading_phi > np.pi]
            
        x = np.c_[x,PRI_lep_phi]
        x = np.c_[x,PRI_met_phi]
        if i > 1:
            x = np.c_[x,PRI_jet_leading_phi]
        if i > 3:
            x = np.c_[x,PRI_jet_subleading_phi]

        #-----------------------------------------------
        PRI_tau_eta = x[:,dic["PRI_tau_eta"]]
        PRI_lep_eta = x[:,dic["PRI_lep_eta"]]
        if i > 1:
            PRI_jet_leading_eta = x[:,dic["PRI_jet_leading_eta"]]
        if i > 3:
            PRI_jet_subleading_eta = x[:,dic["PRI_jet_subleading_eta"]]
            
        # if PRI_tau_eta<0
        index_eta_negative = (PRI_tau_eta<0)
        # the abs is calculated for the following features 
        PRI_tau_eta[index_eta_negative] = np.abs(PRI_tau_eta[index_eta_negative])
        PRI_lep_eta[index_eta_negative] = np.abs(PRI_lep_eta[index_eta_negative])
        if i > 1:
            PRI_jet_leading_eta[index_eta_negative]  = np.abs(PRI_jet_leading_eta[index_eta_negative])
        if i > 3:
            PRI_jet_subleading_eta[index_eta_negative]  = np.abs(PRI_jet_subleading_eta[index_eta_negative])

        x = np.c_[x,PRI_tau_eta]
        x = np.c_[x,PRI_lep_eta]
        if i > 1:
            x = np.c_[x,PRI_jet_leading_eta]
        if i > 3:
            x = np.c_[x,PRI_jet_subleading_eta]

        # removing the replaced featurs
        x = np.delete(x, dic["PRI_tau_eta"], axis=1)
        x = np.delete(x, dic["PRI_tau_phi"]-1, axis=1)
        x = np.delete(x, dic["PRI_lep_eta"]-2, axis=1)
        x = np.delete(x, dic["PRI_lep_phi"]-3, axis=1)
        x = np.delete(x, dic["PRI_met_phi"]-4, axis=1)
        if i > 1:
            x = np.delete(x, dic["PRI_jet_leading_eta"]-5, axis=1)
            x = np.delete(x, dic["PRI_jet_leading_phi"]-6, axis=1)
        if i > 3:
            x = np.delete(x, dic["PRI_jet_subleading_eta"]-7, axis=1)
            x = np.delete(x, dic["PRI_jet_subleading_phi"]-8, axis=1)
            
        results.append(x)
    return results
