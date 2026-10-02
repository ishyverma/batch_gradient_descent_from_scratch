
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(train_path, target_column="SalePrice", id_column="Id"):

    df = pd.read_csv(train_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    y = df[target_column].copy()

    columns_to_drop = [target_column]

    if id_column in df.columns:
        columns_to_drop.append(id_column)

    X = df.drop(columns=columns_to_drop)

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


def build_preprocessor(X):

    numerical_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Missing"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor


def prepare_data(
    train_path,
    test_size=0.2,
    random_state=42
):

    X, y = load_data(train_path)

    X_train, X_val, y_train, y_val = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)

    X_val_processed = preprocessor.transform(X_val)

    feature_names = preprocessor.get_feature_names_out()

    return {
        "X_train": np.asarray(X_train_processed, dtype=float),
        "X_val": np.asarray(X_val_processed, dtype=float),
        "y_train": np.asarray(y_train, dtype=float),
        "y_val": np.asarray(y_val, dtype=float),
        "feature_names": feature_names,
        "preprocessor": preprocessor
    }
