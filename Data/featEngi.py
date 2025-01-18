import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the CSV file into a DataFrame
df = pd.read_csv('Trdata1.0Feat.csv')

# Perform Label Encoding
label_encoder = LabelEncoder()

# Assume 'category_column' is the column to be label encoded
df['gender'] = label_encoder.fit_transform(df['gender'])
df['work_type'] = label_encoder.fit_transform(df['work_type'])
df['smoking_status'] = label_encoder.fit_transform(df['smoking_status'])


# Perform One-Hot Encoding
# Assume 'another_category_column' is the column to be one-hot encoded
df = pd.get_dummies(df, columns=['ever_married'])
df = pd.get_dummies(df, columns=['Residence_type'])

# Save the transformed DataFrame to a new CSV file
df.to_csv('FeaturengineeredData.csv', index=False)







