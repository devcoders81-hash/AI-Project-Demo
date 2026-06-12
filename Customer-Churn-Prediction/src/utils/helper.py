import os
import joblib
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
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

    def evaluate_models(self, X_train, y_train, X_test, y_test, models,params):

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
            train_acc_score=accuracy_score(y_pred_train,y_train)
            train_cr_score=classification_report(y_pred_train,y_train)
            train_com_score=confusion_matrix(y_pred_train,y_train)
            test_acc_score=accuracy_score(y_pred,y_test)
            test_cr_score=classification_report(y_pred,y_test)
            test_com_score=confusion_matrix(y_pred,y_test)
            print(f"the training score is accuracy is: {train_acc_score} : Classification report metrics is:{train_cr_score} : confusion matrix is :{train_com_score}")
            print(f"the testing score is accuracy is: {test_acc_score} : Classification report metrics is:{test_cr_score} : confusion matrix is :{test_com_score}")
            report[model_name] = test_acc_score

            trained_models[model_name] = model

        return report, trained_models