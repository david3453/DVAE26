import pandas as pd
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
from huggingface_hub import Repository
from huggingface_hub import HfApi
from huggingface_hub import login

import pickle
#import automl
import flaml
from flaml import AutoML
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score


modelname = "strokepredictorModel"

df = pd.read_csv(r'../Data/TrainingData2.0.csv')

print("\n\n")

X = df.drop(columns=['stroke'])
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


login()

with open("F1WeightedModel.pkl", "rb") as f:
    automl = pickle.load(f)


api = HfApi()
api.create_repo(repo_id="FinalProjectRepo", exist_ok=True, private=False)
#automl.push_to_hub("MyLab5RepoV3")
automl.push_to_hub("MyLab5RepoV3", model_id="F1WeightedModel")
print("\n\n\n")




#print(X_test.iloc[0])
#print(X_test.iloc[0][11])
#print(X_test.iloc[0][12])


#pred = automl.predict(X_test)
