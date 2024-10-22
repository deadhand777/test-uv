import argparse
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
from feature_engine.creation import MathFeatures
from feature_engine.transformation import YeoJohnsonTransformer
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# from src.utils import compress_joblib_model, pd_read_s3_parquet  #, upload_model
# from utils import pd_read_s3_parquet  # , upload_model


def load_data(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads data from a given path and splits it into X and y.

    Args:
        data_path (str): The path to the data file.

    Returns:
        X (pd.DataFrame): The features.
        y (pd.Series): The target.
    """
    if not isinstance(data_path, str):
        raise TypeError("local_model_path must be a string")

    df: pd.DataFrame = pd.read_parquet(data_path, engine="pyarrow")
    X: pd.DataFrame = df.drop("class", axis=1)
    y: pd.Series = df["class"]

    return X, y


if __name__ == "__main__":
    logger.add(
        sink="train.log",
        format="{time:YYYY-MM-DD HH:MM:SS} | {level} | {message}",
        rotation="10 MB",
        retention="1 week",
        compression="zip",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--n_estimators", type=int, default=15)
    parser.add_argument("--random_state", type=int, default=1)
    args = parser.parse_args()

    # Load Data
    logger.info("Loading data...")
    X, y = load_data(args.train)
    logger.info(f"Loaded data: {X.shape}, {y.shape}")

    # Split Data
    logger.info("Processing data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=args.random_state)
    logger.info(f"Split data: train ({X_train.shape}, {y_train.shape}), test ({X_test.shape}, {y_test.shape})")

    # Train Model
    logger.info("Training model...")
    model: sklearn.pipeline.Pipeline = Pipeline(
        steps=[
            (
                "einkommen",
                MathFeatures(
                    variables=[
                        "sepal_len",
                        "sepal_wid",
                    ],
                    func="sum",
                    missing_values="ignore",
                    new_variables_names=["einkommen"],
                ),
            ),
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=0.1),
            ),
            ("scaler", StandardScaler()),
            ("power_transformation", YeoJohnsonTransformer()),
            (
                "model",
                RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state),
            ),
        ]
    )
    model.fit(X_train, y_train)
    logger.info("Trained model")

    # Evaluate Model
    logger.info("Evaluating model...")
    y_pred: np.ndarray = model.predict(X_test)
    accuracy: float = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy}")

    # Save Model
    logger.info("Saving model...")
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    logger.info("Saved model")
