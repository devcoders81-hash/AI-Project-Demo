from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class DataTransformationConfig:
    def __init__(self):
        pass
    
    def data_transformation(self,df):
        X=df.drop('MedHouseVal',axis=1)
        y=df['MedHouseVal']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        sc=StandardScaler()
        X_train_scaled=sc.fit_transform(X_train)
        X_test_scaled=sc.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test