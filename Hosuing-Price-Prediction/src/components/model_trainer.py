from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.utils.helper import Helper

import os


class ModelTrainer:

    def __init__(self):
        self.helper = Helper()
        self.model_path = os.path.join(
            "artifacts",
            "best_model.joblib"
        )

    def initiate_model_trainer(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        models = {

            "Linear Regression": LinearRegression(),

            "Ridge": Ridge(),

            "Lasso": Lasso(),

            "ElasticNet": ElasticNet(),

            "Decision Tree": DecisionTreeRegressor(),

            "Random Forest": RandomForestRegressor(
                random_state=42
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42
            ),

            "AdaBoost": AdaBoostRegressor(
                random_state=42
            ),

            "XGBoost": XGBRegressor(
                random_state=42
            )
        }

        params = {

            "Linear Regression": {},

            "Ridge": {
                "alpha": [0.01, 0.1, 1, 10, 100]
            },

            "Lasso": {
                "alpha": [0.01, 0.1, 1, 10, 100]
            },

            "ElasticNet": {
                "alpha": [0.01, 0.1, 1, 10],
                "l1_ratio": [0.2, 0.5, 0.8]
            },

            "Decision Tree": {
                "criterion": [
                    "squared_error",
                    "absolute_error"
                ],
                "max_depth": [5, 10, 15]
            },

            "Random Forest": {
                "n_estimators": [50, 100],               
                "max_depth":[10,15],
                "min_samples_split":[5,10],
                "min_samples_leaf":[2,4]
            },

            "Gradient Boosting": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1]
            },

            "AdaBoost": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1]
            },

            "XGBoost": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }
        }

        model_report, trained_models = (
            self.helper.evaluate_models(
                X_train,
                y_train,
                X_test,
                y_test,
                models,
                params
            )
        )
        
        best_model_name = max(
            model_report,
            key=model_report.get
        )

        best_model_score = model_report[
            best_model_name
        ]

        best_model = trained_models[
            best_model_name
        ]

        print(
            f"Best Model : {best_model_name}"
        )
        print(
            f"R2 Score   : {best_model_score}"
        )

        self.helper.save_obj(
            self.model_path,
            best_model
        )

        return best_model_score