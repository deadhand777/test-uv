import io
import os
import tarfile
from typing import Any

import boto3
import pandas as pd
import sagemaker
from loguru import logger


def check_s3_path(s3_path: str) -> None:
    """
    Checks that the given string is a valid S3 path.

    Args:
        - s3_path (str): The S3 path to check.

    Raises:
        - TypeError: If s3_path is not a string.
        - ValueError: If s3_path does not start with 's3://'.
    """
    if not isinstance(s3_path, str):
        raise TypeError("model_path must be a string")

    if not s3_path.startswith("s3://"):
        raise ValueError(f"{s3_path} must start with 's3://'")


def check_string_type(string: str) -> None:
    """
    Checks that the given argument is a string.

    Args:
        - string (str): The string to check.

    Raises:
        - TypeError: If string is not a string.
    """
    if not isinstance(string, str):
        raise TypeError("local_model_path must be a string")


def pd_read_s3_parquet(
    s3_data_path: str,
    **args: Any,
) -> pd.DataFrame:
    """
    Reads a Parquet file from an S3 bucket and returns a Pandas DataFrame.

    Args:
        s3_data_path (str): The S3 location of the Parquet file.
        **args: The keyword arguments to pass to pd.read_parquet.

    Returns:
        pandas.DataFrame: The Parquet data as a Pandas DataFrame.

    Examples:
    >>> s3_path: str = "s3://sagemaker-eu-central-1-763678331342/scikit-learn/demo/data/data.parquet"
    >>> df: pd.DataFrame = pd_read_s3_parquet(s3_data_path=s3_path)
    """

    # Run checks
    check_s3_path(s3_data_path)

    # Get data from S3
    bucket, key_prefix = s3_data_path[5:].split("/", 1)
    s3_client: boto3.client = boto3.client("s3")
    obj: dict = s3_client.get_object(Bucket=bucket, Key=key_prefix)

    # Convert to Pandas DataFrame
    df: pd.DataFrame = pd.read_parquet(io.BytesIO(obj["Body"].read()), **args)

    return df


def upload_data(local_data_path: str, s3_data_path: str, verbose: bool = True) -> None:
    """
    Uploads a local data file to an S3 bucket.

    Args:
        - local_data_path (str): The path to the local data file.
        - s3_data_path (str): The path in the S3 bucket to which the data file should be uploaded.
        - verbose (bool): If True, print the S3 path to which the data was uploaded.

    Returns:
        None

    Examples:
    >>> local_path: str = "data/data.parquet"
    >>> s3_path: str = "s3://sagemaker-eu-central-1-763678331342/scikit-learn/demo/data"
    >>> upload_data(local_data_path = local_path, s3_data_path = s3_path)
    """

    # Run checks
    check_string_type(local_data_path)
    check_s3_path(s3_data_path)

    # Upload data to S3
    bucket, key_prefix = s3_data_path[5:].split("/", 1)
    session: sagemaker.Session = sagemaker.Session()
    session.upload_data(local_data_path, bucket=bucket, key_prefix=key_prefix)

    if verbose:
        print(f"Data uploaded to {s3_data_path}/{local_data_path.split('/')[-1]}")


def upload_model(local_model_path: str, s3_model_path: str, verbose: bool = True) -> None:
    """
    Uploads a local model file to an S3 bucket.

    Args:
    - local_model_path (str): The path to the local model file.
    - s3_model_path (str): The path to the S3 bucket where the model should be uploaded.
    - verbose (bool): If True, print the S3 path to which the data was uploaded.

    Returns:
    - None

    Examples:
    >>> local_path: str = "./script_mode/models/model.tar.gz"
    >>> s3_path: str = "s3://sagemaker-eu-central-1-763678331342/scikit-learn/demo/models"
    >>> upload_model(local_model_path=local_path, s3_model_path=s3_path)
    """

    # Run checks
    check_string_type(local_model_path)
    check_s3_path(s3_model_path)

    # Upload model to S3
    bucket, key_prefix = s3_model_path[5:].split("/", 1)
    session: sagemaker.Session = sagemaker.Session()
    session.upload_data(local_model_path, bucket=bucket, key_prefix=key_prefix)

    if verbose:
        print(f"Model uploaded to {s3_model_path}/{local_model_path.split('/')[-1]}")


def compress_joblib_model(model_path: str, output_path: str | None = None, verbose: bool = True) -> str:
    """
    Compresses a scikit-learn model saved as a .joblib file into a .tar.gz archive.

    Args:
    - model_path (str): The path to the saved .joblib file.
    - output_path (str | None, optional): The path for the output .tar.gz file. If not provided,
                                   it will use the same directory as the model file.
    - verbose (bool): If True, print the S3 path to which the data was uploaded.

    Returns:
    - str: The path to the compressed .tar.gz file.

    Examples:
    >>> compress_joblib_model(model_path="model.joblib")
    """

    # Run checks
    check_string_type(model_path)

    assert os.path.exists(model_path)
    assert model_path.endswith(".joblib")

    # Set the default output path if not provided
    if output_path is None:
        output_path = f"{model_path.removesuffix('.joblib')}.tar.gz"

    # Create a tar.gz archive and add the joblib model file to it
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(model_path, arcname=os.path.basename(model_path))

    if verbose:
        print(f"Model compressed as .tar.gz archive and saved to: {output_path}")

    return output_path


def delete_s3_prefix(s3_data_path: str) -> None:
    """
    Deletes all objects in the specified S3 bucket with the given key prefix.

    Args:
        - s3_data_path (str): The path to the S3 bucket and key prefix.

    Returns:
        None

    Examples:
    >>> s3_path: str = "s3://sagemaker-eu-central-1-763678331342/foo"
    >>> delete_s3_prefix(s3_data_path=s3_path)
    """

    # Run checks
    check_s3_path(s3_data_path)

    # Delete data from S3
    bucket, key_prefix = s3_data_path[5:].split("/", 1)
    s3_client: boto3.client = boto3.resource("s3")
    s3_client.Bucket(bucket).objects.filter(Prefix=key_prefix).delete()
    logger.info(f"Deleted bucket prefix: {bucket}/{key_prefix}")


def delete_s3_objects(
    s3_data_path: str,
) -> None:
    """
    Deletes all objects in the specified S3 bucket with the given key prefix.

    Args:
        - s3_data_path (str): The path to the S3 bucket and key prefix.
        - region_name (str): The AWS region where the S3 bucket is located. Defaults to "eu-central-1".
        - aws_profile_name (str): The name of the AWS profile to use. Defaults to "763678331342_ProgrammaticAccessUser".

    Returns:
        None

    Examples:
    >>> s3_data_path = "s3://sagemaker-eu-central-1-763678331342/foo"
    >>> delete_s3_objects(s3_data_path=s3_data_path)
    """

    # Run checks
    check_s3_path(s3_data_path)

    # Delete data from S3
    bucket, key_prefix = s3_data_path[5:].split("/", 1)
    s3_client: boto3.client = boto3.client("s3")

    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=key_prefix)

    if "Contents" in response:
        objects = [response.get("Contents")[i].get("Key") for i in range(len(response.get("Contents")))]
        logger.info(f"Objects in bucket: {objects}")
        [s3_client.delete_object(Bucket=bucket, Key=obj["Key"]) for obj in response.get("Contents")]
        logger.info(f"Deleted objects in bucket: {bucket}/{key_prefix}")
    else:
        logger.info(f"No objects in bucket: {bucket}/{key_prefix}")
