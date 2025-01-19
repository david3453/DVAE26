import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy import stats

df = pd.read_csv('healthcare-dataset-stroke-data.csv')



df_cleaned = df.dropna(subset=[df.columns[9]])

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('TrainingData1.0.csv', index=False)



