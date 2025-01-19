import pandas as pd
from sklearn.preprocessing import LabelEncoder

#This file was used to feature engineer the data using the sklearn and panda libraries. 
# Load the CSV file into a DataFrame
df = pd.read_csv('TrainingData2.0.csv')
df = df.drop('gender_Other', axis=1)

print(df.head())



# Convert boolean values to integers

# Save the transformed DataFrame to a new CSV file
df.to_csv('TrainingData2.0.csv', index=False)






