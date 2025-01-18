import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the CSV file into a DataFrame
df = pd.read_csv('FeatureengineeredData2.0.csv')

# Perform Label Encoding
#label_encoder = LabelEncoder()

# Specify the columns you want to convert
columns_to_convert = ['gender_Female', 'gender_Male', 'gender_Other']

# Convert boolean values to integers
df[columns_to_convert] = df[columns_to_convert].astype(int)

# Save the transformed DataFrame to a new CSV file
df.to_csv('FeatureengineeredData3.0.csv', index=False)

print("Conversion completed and saved to 'transformed_file.csv'")






