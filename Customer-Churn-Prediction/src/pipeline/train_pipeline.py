from src.components.data_preprocessing import DataPreprocessing
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
data_preprocessing=DataPreprocessing()
data_transformation=DataTransformationConfig()
model_trainer=ModelTrainer()
df=data_preprocessing.data_prepare()
X_train, X_test, y_train, y_test=data_transformation.data_transformation(df)
model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)