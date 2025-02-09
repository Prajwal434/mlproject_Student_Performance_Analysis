import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path : str = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split train and test data input")
            x_train,y_train,x_test,y_test=(
                train_array[:, :-1],
                train_array[:, -1].reshape(-1,1),
                test_array[:, :-1],
                test_array[:, -1].reshape(-1,1)
            )
            models = {
                "Random forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "linear regression" : LinearRegression(),
                "K-Neighbors classifier" : KNeighborsRegressor(),
                "XGB Classifier" : XGBRegressor(),
                "Catboosting classifier" : CatBoostRegressor(verbose=0),
                "adaboost classifier" : AdaBoostRegressor(), 
            }
            model_report:dict=evaluate_models(X_train= x_train,Y_train=y_train,X_test=x_test,Y_test=y_test,Models=models)


            print("Model Report:", model_report)

            if not model_report:
              raise CustomException("Model evaluation failed. No scores found.")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(x_test)

            r2_sqaure = r2_score(y_test, predicted)
            print(f"Final R² Score:, {r2_sqaure:.3f}")
            return r2_sqaure


        except Exception as e:
            raise CustomException(e,sys)