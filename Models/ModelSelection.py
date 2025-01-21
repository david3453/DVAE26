import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
#import automl
import flaml
from flaml import AutoML
import matplotlib.pyplot as plt




df = pd.read_csv(r'C:/DVAE26/Data/TrainingData2.0.csv')
print("\n\n")

X = df.drop(columns=['stroke'])
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Step 4: Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=40)

# Print the shapes of the splits to verify
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

settings_other = {
    "metric": 'f1',
    "log_file_name": "AIModel.log"
}
automl = AutoML()
automl.fit(X_train, y_train, task="classification", time_budget=60, **settings_other)


print("\n\n\n\n")
print(automl.best_estimator)
# lgbm
print("\n\n\n\n")

print(automl.best_config)

print("\n\n\n")

plt.barh(
    automl.model.estimator.feature_name_, automl.model.estimator.feature_importances_
)
