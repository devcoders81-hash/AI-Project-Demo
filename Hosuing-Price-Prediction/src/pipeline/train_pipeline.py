from src.components.data_preprocessing import DataPreprocessing
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer

data_preprocessing=DataPreprocessing()
data_transformation=DataTransformationConfig()
model_trainer=ModelTrainer()
print("Initiate Data Preprocessing")
df=data_preprocessing.data_prepare()
print("Completed Data Preprocessing")
print("Initiate Data Transformation")
X_train, X_test, y_train, y_test=data_transformation.data_transformation(df)
print("Completed Data Transformation")
print("Initiate Model Trainer")
r2_metrics=model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
print("Completed Model Trainer")
print(r2_metrics)
