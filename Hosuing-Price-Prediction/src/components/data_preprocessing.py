from sklearn.datasets import fetch_california_housing
import pandas as pd
class DataPreprocessing:
    def __init__(self):
        self.housing_df = fetch_california_housing()


    def data_prepare(self):
        df=pd.DataFrame(self.housing_df.data, columns=self.housing_df.feature_names)
        df[self.housing_df.target_names[0]] = self.housing_df.target
        return df