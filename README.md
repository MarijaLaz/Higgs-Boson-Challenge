# Project 1 : Higgs Boson Challenge

This is a project realised for the Machine Learning Course [CS-433]. We were working on finding the best model for the Higgs Boson Challenge. After testing 6 models we ended up using Logistic Regression for our final submission.

## Folder Structure

```

.
├── data                               # The train and test data sets
│   ├── test.csv.zip
│   └── train.csv.zip
├── results                            # Final results for submission
│   └── results_final.csv
├── hyperparameterTestResult.txt       # Results for the hyperparameters after crossvalidation
├── Report.pdf                         # The report
├── README.md                          # The ReadMe file
├── run.py                             # Script for training the models with the train set and get predictions for the test set
├── scripts                            # Scripts used for testing the models and analysis of the data
│   ├── dataAnalysisPlot.py
│   ├── testing_parameters.py
│   └── Test_Models.ipynb
└── utils                              # The functions used in the scripts
    ├── crossvalidation.py             # Functions for performing crossvalidation
    ├── features.py                    # Dictionaries containing infromations for the columns
    ├── helpers.py                     # Helping functions
    ├── implementations.py             # Mandatory functions
    ├── loss_gradient.py               # Functions for the computing the loss and the gradients
    └── preprocessing.py               # Functions for feature preprocessing

```


## Zipped Files
Before running the code you will need to unzip the files containing the train and test data:

* data/train.csv.zip     
* data/test.csv.zip 

## Running the code
To run our code and get the final predictions you can use the following command:

```python
python3 run.py
```
## Final submission results

The final results after running the run.py script can be found in the **results** folder under the name *results.csv*

## Authors
* Marija Lazaroska     
* Deborah Scherrer Ma  
* Méline Zhao  
