import os 
import sys
from dataclasses import dataclass

import os
import warnings
from sklearn.exceptions import ConvergenceWarning

# Silence convergence warnings from models like LogisticRegression
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Silence joblib CPU warning
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # or however many logical cores your system has


from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split train and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Linear Regression": LinearRegression(),
                "XGB Classifier": XGBRegressor(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            } 

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test,
                                                models=models)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model  = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both train and test data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            logging.info(f"Trained model score on test data")
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)
        