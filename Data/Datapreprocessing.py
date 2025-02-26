import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy import stats
import Outlierdetection

def preprocess_data(dataset, savename):

    
    df = dataset

    #Initial data cleaning
    df_processed = df.dropna(subset=[df.columns[9]]) ## Remove all 0 BMI values
    df_processed = df_processed.loc[df.iloc[:, 0] != 56156] ##Remove row with "other" in gender
    df_processed = df_processed.drop(columns=['id']) ## Remove id column
   

    ##Outlier detection
    df_processed = Outlierdetection.RemoveOutliers(df_processed)

    # One hot encoding columns
    df_processed = pd.get_dummies(df_processed, columns=['work_type'], drop_first=True)

    # Label encoding columns
    encoder = LabelEncoder()
    df_processed['smoking_status'] = encoder.fit_transform(df_processed['smoking_status'])

    # Convert to 0 or 1
    df_processed['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df_processed['ever_married'] = df_processed['ever_married'].map({'Yes': 1, 'No': 0})
    df_processed['Residence_type'] = df_processed['Residence_type'].map({'Urban': 1, 'Rural': 0})
    df_processed['work_type_Never_worked'] = df_processed['work_type_Never_worked'].astype(int)
    df_processed['work_type_Private'] = df_processed['work_type_Private'].astype(int)
    df_processed['work_type_Self-employed'] = df_processed['work_type_Self-employed'].astype(int)
    df_processed['work_type_children'] = df_processed['work_type_children'].astype(int)
    df_processed['gender'] = df_processed['gender'].astype(int)

    ##Save data to csv and return it aswell
    df_processed.reset_index(drop=True, inplace=True)
    savename = savename + '.csv'
    df_processed.to_csv(savename, index=False)
    return df_processed




