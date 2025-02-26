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

def CreateModel(NameModel, ClassWeightSign, Weight0, Weight1,
                Metric,csvdataset):
    ##df = pd.read_csv(r'../Data/TrainingData2.0.csv')
    df = csvdataset
    print("\n\n")
    
    X = df.drop(columns=['stroke'])
    y = df['stroke']

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=40)
    classweightsign = 0
    weight0 = 0
    weight1 = 0
    modelname = NameModel
    if not modelname.endswith('.pkl'):
        modelname += '.pkl'
    if ClassWeightSign == 1:
        classweightsign = 1
        weight0 = Weight0
        weight1 = Weight1
    ##Default metric is f1
    metric = Metric


    ##class_weights = {0: 1, 1: 17}  # Assuming 1 is the positive class
    ##sample_weightArr = np.array([class_weights[y] for y in y_train])

    # Step 4: Further split the training set into training and validation sets

    # Print the shapes of the splits to verify
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Testing set shape:", X_test.shape, y_test.shape)

    log_file_name = NameModel + '.log'

    settings_other = {
        "metric": metric,  # Pass the custom recall metric function
        "log_file_name": log_file_name
    }
    automl = AutoML()
    #automl.fit(X_train, y_train, task="classification", time_budget=60, **settings_other)
    if classweightsign == 1:
        print("Weighted model created")
        class_weights = {0: weight0, 1: weight1}  # Assuming 1 is the positive class
        sample_weightArr = np.array([class_weights[y] for y in y_train])
        automl.fit(X_train, y_train, task="classification", sample_weight=sample_weightArr, time_budget=60, **settings_other)
    else:
        automl.fit(X_train, y_train, task="classification", time_budget=60, **settings_other)

    print("\n\n\n\n")
    print(automl.best_estimator)
    # lgbm
    print("\n\n\n\n")

    print(automl.best_config)

    print("\n\n\n")

    try:
        # Attempt to use feature_name_ if it exists
        feature_names = automl.model.estimator.feature_name_
    except AttributeError:
        # Fallback to feature_names_in_ if feature_name_ does not exist
        feature_names = automl.model.estimator.feature_names_in_

    plt.barh(feature_names, automl.model.estimator.feature_importances_)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    # Display the plot
    plt.show()

    with open(modelname, 'wb') as model_file:
        pickle.dump(automl, model_file)

    print("Model saved successfully!")
    return automl



