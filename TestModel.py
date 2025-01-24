import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
#import automl
import flaml
from flaml import AutoML
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, recall_score



df = pd.read_csv(r'../Data/TrainingData2.0.csv')

X = df.drop(columns=['stroke'])
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

with open('F1WeightedModel.pkl', 'rb') as model_file:
    loaded_automl = pickle.load(model_file)

y_pred = loaded_automl.predict(X_test)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)
print("X_test size:", X_test.shape)

print(f"\n\n\nF1 Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")
# Step 2: Calculate precision
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.2f}")

# Step 3: Calculate recall
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.2f}")
print("Confusion Matrix:")
print(conf_matrix)