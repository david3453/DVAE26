import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy import stats

df = pd.read_csv('healthcare-dataset-stroke-data.csv')


#Remove outlier in gender and remove all rows with N/A in bmi
df_processed = df.dropna(subset=[df.columns[9]])
df_processed = df_processed[df.iloc[:, 0] != 56156]

##One hot encoding columns

df_processed = pd.get_dummies(df_processed, columns=['work_type'], drop_first=True)
 
#Label encoding columns

encoder = LabelEncoder()
df_processed['smoking_status'] = encoder.fit_transform(df_processed['smoking_status'])


#Convert to 0 or 1
df_processed['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df_processed['ever_married'] = df_processed['ever_married'].map({'Yes': 1, 'No': 0})
df_processed['Residence_type'] = df_processed['Residence_type'].map({'Urban': 1, 'Rural': 0})
df_processed['work_type_Never_worked'] = df_processed['work_type_Never_worked'].astype(int)
df_processed['work_type_Private'] = df_processed['work_type_Private'].astype(int)
df_processed['work_type_Self-employed'] = df_processed['work_type_Self-employed'].astype(int)
df_processed['work_type_children'] = df_processed['work_type_children'].astype(int)




print(df_processed.head())

# Save the cleaned DataFrame to a new CSV file
df_processed.to_csv('TrainingData1.0.csv', index=False)




