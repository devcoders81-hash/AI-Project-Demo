import os
import joblib
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import GridSearchCV
class Helper:
    def __init__(self):
        pass

    def save_obj(self,file_path,obj):
        try:
            dir_path=os.path.dirname(file_path)
            os.makedirs(dir_path,exist_ok=True)
            with open(file_path,"wb") as file_obj:
                joblib.dump(obj,file_obj)
        except Exception as ex:
            raise Exception(ex)

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):

        report = {}
        trained_models = {}

        for model_name, model in models.items():

            param_grid = params.get(model_name, {})
            print(f"the model name is : {model_name}")
            if param_grid:
                gs = GridSearchCV(
                    model,
                    param_grid,
                    cv=3,
                    n_jobs=-1
                )

                gs.fit(X_train, y_train)

                model = gs.best_estimator_

            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            train_score=r2_score(y_pred_train,y_train)
            test_score=r2_score(y_pred,y_test)
            print(f"the training score is : {train_score}")
            print(f"the testing score is : {test_score}")
            report[model_name] = test_score

            trained_models[model_name] = model

        return report, trained_models