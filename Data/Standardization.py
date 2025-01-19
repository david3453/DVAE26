import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy import stats

df = pd.read_csv('TrainingData2.1.csv')

columns_to_check = ['age', 'bmi', 'avg_glucose_level']  # Replace with your actual column names


#Calculate Z-scores for the specified columns
z_scores = df[columns_to_check].apply(stats.zscore)

#Add Z-score columns to the DataFrame
for col in columns_to_check:
    df[f'{col}_zscore'] = z_scores[col]

#Display the DataFrame with Z-scores
print("\nDataFrame with Z-scores:")
print(df.head())

#Identify outliers for each column based on Z-scores (e.g., Z-score > 3 or < -3)
outliers = df[(z_scores.abs() > 3).any(axis=1)]

#Print the outliers
print("\nOutliers in the specified columns:")
print(outliers)







#print(df.head())
# Save the transformed DataFrame to a new CSV file
df.to_csv('TrainingData3.0TESTRUN.csv', index=False)