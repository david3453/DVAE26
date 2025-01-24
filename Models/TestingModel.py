import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
import pickle
#import automl
import flaml
from flaml import AutoML
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

df = pd.read_csv(r'../Data/TrainingData2.0.csv')

print("\n\n")

X = df.drop(columns=['stroke'])
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


with open("automlF1Model.pkl", "rb") as f:
    automl = pickle.load(f)
pred = automl.predict(X_test)


# Predict the labels for the test set
y_pred = automl.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)
print(f"\n\nAccuracy: {accuracy}\n")
print(f"F1 Score: {f1_score}\n")

