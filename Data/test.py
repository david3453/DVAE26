import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
#import automl
import flaml
from flaml import AutoML
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import recall_score, make_scorer



df = pd.read_csv(r'../Data/TrainingData2.0.csv')



def detect_columnss(df, threshold=15):
    columns_with_many_unique_values = [col for col in df.columns if df[col].nunique() > threshold]
    return columns_with_many_unique_values

print(detect_columnss(df))