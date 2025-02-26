import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy import stats

def detect_columnss(df, threshold=15):
    columns_with_many_unique_values = [col for col in df.columns if df[col].nunique() > threshold]
    return columns_with_many_unique_values

def RemoveOutliers(dataset):
    # Load the data from the CSV file
    df = dataset
    ##df = pd.read_csv('TrainingData1.1.csv')

    columns_to_check = detect_columnss(df)  # Replace with your actual column names
    ##columns_to_check = ['age', 'bmi', 'avg_glucose_level']  # Replace with your actual column names

    # Calculate Z-scores for the specified columns
    z_scores = df[columns_to_check].apply(stats.zscore)

    # Add Z-score columns to the DataFrame
    for col in columns_to_check:
        df[f'{col}_zscore'] = z_scores[col]

    # Display the DataFrame with Z-scores
    ##print("\nDataFrame with Z-scores:")
    ##print(df.head())

    # Identify outliers for each column based on Z-scores (e.g., Z-score > 3 or < -3)
    outliers = df[(z_scores.abs() > 3).any(axis=1)]

    # Print the outliers
    ##print("\nOutliers in the specified columns:")
    ##print(outliers)

    df_cleaned = df[(z_scores.abs() <= 3).all(axis=1)]

    # Print the cleaned DataFrame

    #print("\nDataFrame after removing outliers:")
    #print(df_cleaned.head())

    df_cleaned = df_cleaned.drop('age_zscore', axis=1)
    df_cleaned = df_cleaned.drop('bmi_zscore', axis=1)
    df_cleaned = df_cleaned.drop('avg_glucose_level_zscore', axis=1)

    #print(df_cleaned.head())

    # Save the transformed DataFrame to a new CSV file
    return df_cleaned
    ##df_cleaned.to_csv('TrainingData2.0.csv', index=False)


