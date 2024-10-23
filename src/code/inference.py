import argparse
import json
import os
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd

# import sklearn
from loguru import logger
from pydantic import BaseModel, ValidationError
from sklearn.pipeline import Pipeline as sklearn_pipeline


class Data(BaseModel):
    sepal_len: float
    sepal_wid: Optional[float] = None
    petal_len: Optional[float] = None
    petal_wid: Optional[float] = None


def model_fn(model_path: str) -> object:
    """
    Deserialize fitted model. Loads a saved model from a file in the model directory.

    Parameters:
    model_path (str): The path to the directory containing the saved model.

    Returns:
    model (object): The loaded model object.
    """
    model = joblib.load(os.path.join(model_path, "model.joblib"))
    return model


def input_fn(request_body: str, content_type: str) -> pd.DataFrame:
    """
    Deserialize input data into a format that can be used by the model.

    Parameters:
    request_body (str): The body of the request sent to the model.
    content_type (str): The content type of the request.

    Returns:
    pd.DataFrame: The input data in a format that can be used by the model.
    """
    expected_features = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]

    if content_type == "application/json":
        # Extract features from JSON key-value pairs
        data = json.loads(request_body)

        # Data validation and processing
        try:
            data_valid = Data(**data)
            logger.info("Data is valid: %s", data_valid.model_dump_json())

            complete_data = {feature: data_valid.model_dump().get(feature, np.nan) for feature in expected_features}
            df = pd.DataFrame.from_dict(complete_data, orient="index").T
            return df

        except ValidationError as e:
            logger.error(e.errors())
            raise ValueError("Validation error in input data") from e
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: pd.DataFrame, model: sklearn_pipeline) -> dict[str, Union[str, int, np.ndarray]]:
    """
    Make predictions against input_data using the model provided.

    Args:
        input_data (pd.DataFrame): The input data to make predictions against.
        model (sklearn.pipeline.Pipeline): The model to use for prediction.

    Returns:
        dict: A dictionary with two keys, 'predicted_class' and 'predicted_probabilities'.
                'predicted_class' is the predicted class, and 'predicted_probabilities' is the
                probability distribution of all classes.
    """

    # Collect predictions
    predicted_class: int = model.predict(input_data)[0]
    predicted_proba: np.ndarray = model.predict_proba(input_data)[0]

    # Return predictions
    result: dict[str, Union[str, int, np.ndarray]] = {
        "predicted_class": predicted_class,
        "predicted_probabilities": predicted_proba.tolist(),
    }
    return result


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data", type=str)
    argparser.add_argument("--model_path", type=str)
    args = argparser.parse_args()

    predict_fn(args.data, args.model_path)
