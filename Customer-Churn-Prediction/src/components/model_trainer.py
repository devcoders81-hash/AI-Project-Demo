from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils.helper import Helper

import os


class ModelTrainer:

    def __init__(self):
        self.helper = Helper()
        self.model_path = os.path.join(
            "artifacts",
            "model/best_model.joblib"
        )

    def initiate_model_trainer(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        models = {
            "Logistic Regression":LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),

            "Random Forest": RandomForestClassifier(
                random_state=42
            ),
            "XGBoost": XGBClassifier(
                random_state=42
            )
        }
        params = {

            "Logistic Regression": {},

            "Decision Tree": {
                "max_depth": [5, 10, 15]
            },

            "Random Forest": {
                "n_estimators": [50, 100],               
                "max_depth":[10,15],
                "min_samples_split":[5,10]
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
            f"Accuracy Score   : {best_model_score}"
        )

        self.helper.save_obj(
            self.model_path,
            best_model
        )

        return best_model_score