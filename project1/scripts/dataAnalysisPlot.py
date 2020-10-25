# Useful starting lines
import numpy as np
from utils.helpers import *
from utils.preprocessing import *
import matplotlib.pyplot as plt

#Load Data
DATA_TRAIN_PATH = '../data/train.csv'
DATA_TEST_PATH  = '../data/test.csv'
y_train, x_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
print("Train data loaded")
y_test, x_test, ids_test = load_csv_data(DATA_TEST_PATH)
print("Test data loaded")

#Create boolean array for train and test data according to jet_num
jet_tr_0 = x_train[:,22] == 0
jet_tr_1 = x_train[:,22] == 1
jet_tr_2 = x_train[:,22] == 2
jet_tr_3 = x_train[:,22] == 3

jet_te_0 = x_test[:,22] == 0
jet_te_1 = x_test[:,22] == 1
jet_te_2 = x_test[:,22] == 2
jet_te_3 = x_test[:,22] == 3
print("Jets created")

#Count number of NaN values in train and test data
nb_nan_jet_tr_0 = np.count_nonzero(x_train[jet_tr_0] == -999)
nb_nan_jet_tr_1 = np.count_nonzero(x_train[jet_tr_1] == -999)
nb_nan_jet_tr_2 = np.count_nonzero(x_train[jet_tr_2] == -999)
nb_nan_jet_tr_3 = np.count_nonzero(x_train[jet_tr_3] == -999)

nb_nan_jet_te_0 = np.count_nonzero(x_test[jet_te_0] == -999)
nb_nan_jet_te_1 = np.count_nonzero(x_test[jet_te_1] == -999)
nb_nan_jet_te_2 = np.count_nonzero(x_test[jet_te_2] == -999)
nb_nan_jet_te_3 = np.count_nonzero(x_test[jet_te_3] == -999)
print("Zeros counted")

#Remove columns with only Nan values (cleaning)
jet_0_tr_clean = removeNaN(x_train[jet_tr_0])
jet_1_tr_clean = removeNaN(x_train[jet_tr_1])
jet_2_tr_clean = removeNaN(x_train[jet_tr_2])
jet_3_tr_clean = removeNaN(x_train[jet_tr_3])

jet_0_te_clean = removeNaN(x_test[jet_te_0])
jet_1_te_clean = removeNaN(x_test[jet_te_1])
jet_2_te_clean = removeNaN(x_test[jet_te_2])
jet_3_te_clean = removeNaN(x_test[jet_te_3])
print("Clean data created")

#Count remaining columns with NaN values
nb_nan_jet_tr_0_clean = np.count_nonzero(jet_0_tr_clean == -999)
nb_nan_jet_tr_1_clean = np.count_nonzero(jet_1_tr_clean == -999)
nb_nan_jet_tr_2_clean = np.count_nonzero(jet_2_tr_clean == -999)
nb_nan_jet_tr_3_clean = np.count_nonzero(jet_3_tr_clean == -999)

nb_nan_jet_te_0_clean = np.count_nonzero(jet_0_te_clean == -999)
nb_nan_jet_te_1_clean = np.count_nonzero(jet_1_te_clean == -999)
nb_nan_jet_te_2_clean = np.count_nonzero(jet_2_te_clean == -999)
nb_nan_jet_te_3_clean = np.count_nonzero(jet_3_te_clean == -999)

#Count number of backgroung/signal per jet_num
bg_jet_0 = np.count_nonzero(y_train[jet_tr_0] == -1)
bg_jet_1 = np.count_nonzero(y_train[jet_tr_1] == -1)
bg_jet_2 = np.count_nonzero(y_train[jet_tr_2] == -1)
bg_jet_3 = np.count_nonzero(y_train[jet_tr_3] == -1)

s_jet_0 = np.count_nonzero(y_train[jet_tr_0] == 1)
s_jet_1 = np.count_nonzero(y_train[jet_tr_1] == 1)
s_jet_2 = np.count_nonzero(y_train[jet_tr_2] == 1)
s_jet_3 = np.count_nonzero(y_train[jet_tr_3] == 1)


#Plotting 
width = 0.4
x_pos_1 = np.arange(4)-width/2
x_pos_2 = np.arange(4)+width/2
x_pos = np.arange(4)

#Subplots for NaN values before and after cleaning
plt.subplot(121)
ay1 = np.array([nb_nan_jet_tr_0,nb_nan_jet_tr_1,nb_nan_jet_tr_2,nb_nan_jet_tr_3])
p1 = plt.bar(x_pos_1,ay1/100000,width, color = 'tab:gray')
ay2 = np.array([nb_nan_jet_te_0,nb_nan_jet_te_1,nb_nan_jet_te_2,nb_nan_jet_te_3])
p2 = plt.bar(x_pos_2, ay2/100000, width, color = 'tab:olive')
plt.title('NaN values in original data')
plt.ylabel('Counts of NaN [x100000]')
plt.xticks(x_pos, ('jet_0', 'jet_1', 'jet_2', 'jet_3'))
plt.legend((p1[0], p2[0]), ('Train data', 'Test data'))

plt.subplot(122)
ay3 = np.array([nb_nan_jet_tr_0_clean,nb_nan_jet_tr_1_clean,nb_nan_jet_tr_2_clean,nb_nan_jet_tr_3_clean])           
p3 = plt.bar(x_pos_1, ay3/1000, width, color='tab:gray')
ay4 = np.array([nb_nan_jet_te_0_clean,nb_nan_jet_te_1_clean,nb_nan_jet_te_2_clean,nb_nan_jet_te_3_clean])
p4 = plt.bar(x_pos_2, ay4/1000, width, color='tab:olive')
plt.title('NaN values in cleaned data')
plt.ylabel('Counts of NaN [x1000]')
plt.legend((p3[0], p4[0]), ('Train data', 'Test data'))
plt.xticks(x_pos, ('jet_0', 'jet_1', 'jet_2', 'jet_3'))
plt.subplots_adjust(wspace = 0.4)
plt.savefig('jet_and_nan.png', dpi=300, bbox_inches='tight')
plt.show()

#Plot for background and signal according to jet_num
plt.close('all')
ay5 = np.array([bg_jet_0,bg_jet_1,bg_jet_2,bg_jet_3])
ay6 = np.array([s_jet_0,s_jet_1,s_jet_2,s_jet_3])
p5 = plt.bar(x_pos, ay5 ,width, color = 'tab:gray')
p6 = plt.bar(x_pos,ay6, width, bottom = ay5,  color = 'tab:olive')
plt.title('Background x Signal values in train data')
plt.xticks(x_pos, ('jet_0', 'jet_1', 'jet_2', 'jet_3'))
plt.ylabel('y labels')
plt.legend((p5[0], p6[0]), ('Background', 'Signal'))
plt.show()
plt.savefig('ylabels.png', dpi=300, bbox_inches='tight')


#Printing values
text = np.array(["Number NaN in train: ","Number NaN in test: ","Number NaN in clean train: ","Number NaN in clean test: ","Number Background: ","Number Signal: "])
ay = [ay1, ay2, ay3, ay4, ay5, ay6]
for i in range(6):
    print("{t} {val}".format(t=text[i],val=ay[i]))

print("\nPercentage background rounded: ", np.round(ay5*100/(ay5+ay6),2))
print("Percentage signal rounded: ", np.round(ay6*100/(ay5+ay6),2))
