import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy import stats

df = pd.read_csv('TrainingData2.0.csv')
df2 = pd.read_csv('healthcare-dataset-stroke-data.csv')


# Display the first few rows of the original DataFrames
print("Original df DataFrame:")
print(df.head())

print("\nOriginal df2 DataFrame:")
print(df2.head())
print("\n\n\n")
# Ensure both DataFrames have the same length

    # Specify the column to copy from df2 and insert into df
column_name = 'avg_glucose_level'  # Replace 'age' with the actual column name you want to copy

index = 0
index2 = 0

# Get the total number of rows in the DataFrame
total_rows = len(df)
total_rows2 = len(df2)

while index < total_rows:

    
    index2 = 0

    while index2 < total_rows2:

        
        #print(f"Comparing ages {df.iloc[index,0]}, {df2.iloc[index2,2]} and {df.iloc[index,1]} , {df2.iloc[index2,3]} and {df.iloc[index,2]}, {df2.iloc[index2,4]} and {df.iloc[index,5]}, {df2.iloc[index2,9]}\n")
        if(df.iloc[index,0] == df2.iloc[index2,2] and df.iloc[index,1] == df2.iloc[index2,3] and df.iloc[index,2] == df2.iloc[index2,4] and df.iloc[index,5] == df2.iloc[index2,9]):
            df.iloc[index,4] = df2.iloc[index2,8]
            #print("CHANGED A VALUE!\n\n\n\n\n\n")
            break
            
        else:
            index2 = index2 + 1
            if(index2 == total_rows2):
                print(f"Failed to find a value at index {index}\n\n")
    
    index = index + 1
    #print(f"Moving forward to index {index}\n")



    df.to_csv('TrainingData2.1.csv', index=False)