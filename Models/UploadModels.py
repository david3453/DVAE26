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

def UploadModel(pkl_filename):
    login()

    repo_id = "David991199/FinalProjectRepo"
    filename = pkl_filename

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=pkl_filename,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
        )
        return True
    except Exception as e:
        print(f"An error occurred while uploading the model: {e}")
        return False






