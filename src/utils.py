import os
import sys
import dill


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)



def evaluate_models(X_train, Y_train, X_test, Y_test, Models):
    best_models = {}
    model_scores = {}

    # Define hyperparameter grids
    param_grids = {
        "Random forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
        "Decision Tree": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
        "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
        "linear regression": {},
        "K-Neighbors classifier": {"n_neighbors": [3, 5, 7]},
        "XGB Classifier": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
        "Catboosting classifier": {"depth": [6, 8], "learning_rate": [0.01, 0.1]},
        "adaboost classifier": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
    }

    for model_name, model in Models.items():
        print(f"Tuning Hyperparameters for {model_name}...")

        if param_grids.get(model_name):
            grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring="r2", n_jobs=-1)
            grid_search.fit(X_train, Y_train.ravel())  # Apply ravel() for 1D array
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Best Params for {model_name}: {best_params}")
        else:
            best_model = model.fit(X_train, Y_train.ravel())

        y_pred = best_model.predict(X_test)
        r2 = r2_score(Y_test, y_pred)

        model_scores[model_name] = r2
        best_models[model_name] = best_model

    return model_scores, best_models

       
