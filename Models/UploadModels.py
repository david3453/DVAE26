import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import pickle
#import automl
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

with open("F1WeightedModel.pkl", "rb") as f:
    myml = pickle.load(f)

print(X_test.iloc[0])
print(X_test.iloc[0,11])
print(X_test.iloc[0,12])
print(X_test.iloc[0,6])



for i in range(10):
    first_row = X_test.iloc[[i]]  # Use double brackets to keep it as a DataFrame with feature names
    predicted = myml.predict(first_row)
    print("I is ",i ," Predicted: ", predicted[0], "\n")
    print("Correct prediction was ", y_test.iloc[i])

print(X_test.iloc[2])
print(X_test.iloc[4])
print(X_test.iloc[7])
print(X_test.iloc[9])




#pred = automl.predict(X_test)
