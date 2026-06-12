from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import joblib
import os

class DataTransformationConfig:

    def __init__(self):
        self.preprocessor_path = os.path.join(
            "artifacts",
            "preprocessor.joblib"
        )

    def data_transformation(self, df):

        df = df.dropna(subset=["Churn"])

        X = df.drop(columns=["Churn", "CustomerID"],axis=1)
        y = df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        numerical_cols = X.select_dtypes(
            exclude="object"
        ).columns

        categorical_cols = X.select_dtypes(
            include="object"
        ).columns

        num_pipeline = Pipeline(
            steps=[
                ("imputer",
                 SimpleImputer(strategy="median"))
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer",
                 SimpleImputer(strategy="most_frequent")),

                ("encoder",
                 OrdinalEncoder())
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num_pipeline",
                    num_pipeline,
                    numerical_cols
                ),
                (
                    "cat_pipeline",
                    cat_pipeline,
                    categorical_cols
                )
            ]
        )

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        os.makedirs("artifacts", exist_ok=True)

        joblib.dump(
            preprocessor,
            self.preprocessor_path
        )

        return X_train, X_test, y_train, y_test