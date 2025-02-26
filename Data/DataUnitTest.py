import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os
from scipy import stats
#import automl
import flaml
from flaml import AutoML
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import recall_score, make_scorer

import Datapreprocessing
import Outlierdetection

##Import data
filepath = '../Data/healthcare-dataset-stroke-data.csv' ##Change this to your liking
TotalData = pd.read_csv(filepath)

##Default name
DataName = 'TrainingData3.0'

##Process the data
ProcessedData = Datapreprocessing.preprocess_data(TotalData,DataName)

##Check if processed data csv file was created
DataCSV = DataName + '.csv'
if os.path.exists(DataCSV):
    print(f"CSV file '{DataCSV}' has been created successfully.")
    CSVData = pd.read_csv(DataCSV)
    ## Compare the first 10 rows of ProcessedData and CSVData
    if ProcessedData.head(10).equals(CSVData.head(10)):
        print("The first 10 rows of ProcessedData and CSVData are equal.")
    else:
        print("The first 10 rows of ProcessedData and CSVData are not equal.")
        

else:
    print(f"CSV file '{DataCSV}' has not been created.")







