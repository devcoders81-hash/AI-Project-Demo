import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
class DataPreprocessing:
    def __init__(self):
        pass
    
    def data_prepare(self):
        testing_df=pd.read_csv("artifacts\data\customer_churn_dataset-testing-master.csv")
        training_df=pd.read_csv("artifacts\data\customer_churn_dataset-training-master.csv")
        df=pd.concat([training_df,testing_df],axis=0, ignore_index=True)
        return df