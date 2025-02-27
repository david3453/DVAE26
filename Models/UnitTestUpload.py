import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from huggingface_hub import hf_hub_download, HfApi, login
import pickle
from flaml import AutoML
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
import UploadModels


modelpklname = 'F1WeightedModel.pkl'

##Test if pkl has been created
with open(modelpklname, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

booleanfunc = UploadModels.UploadModel(modelpklname)

if booleanfunc == True:
    print("Model uploaded successfully.")
else:
    print("Model failed to upload.")



