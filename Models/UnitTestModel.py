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
import ModelSelectionFile

df = pd.read_csv(r'../Data/TrainingData3.0.csv')
modelmetric = 'f1'
modelname = 'F1WeightedModel'
modelclassweightsign = 1
modelweight0 = 1
modelweight1 = 17



TrainedModel = ModelSelectionFile.CreateModel(modelname, modelclassweightsign,
                                               modelweight0, modelweight1,
                                                 modelmetric, df)


##Test if pkl has been created
modelpklname = modelname + '.pkl'
with open(modelpklname, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

print(f"Pickle file '{modelpklname}' has been created successfully.")

##Test if log file has been created
modelLogname = modelname + '.log'

if os.path.exists(modelLogname):
    print(f"Log file '{modelLogname}' has been created successfully.")
else:
    print(f"Log file '{modelLogname}' has not been created.")


##Test if loaded model predicts the same as the returned model.

errorpredict = False
for i in range(10):
    first_row = df.iloc[[i]]
    predicted = TrainedModel.predict(first_row)
    predicted2 = loaded_model.predict(first_row)
    if predicted[0] != predicted2[0]:
        errorpredict = True
            
if errorpredict == True:
    print("Error: Predictions do not match.")
else:
    print("Predictions match.")






