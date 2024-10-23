import datetime
from typing import Optional

import sagemaker
from loguru import logger
from sagemaker.serializers import JSONSerializer
from sagemaker.sklearn import SKLearn
from sagemaker.sklearn.model import SKLearnModel

from src.code.utils import check_bool_type, check_s3_path, check_string_type


def deploy_sklearn_model(
    name: str,
    role: str,
    image_uri: str,
    script_mode: bool = False,
    training_input_path: Optional[str] = None,
    model_data: Optional[str] = None,
    entry_point: str = "train.py",
    source_dir: str = "./src/code",
    hyperparameters: Optional[dict[str, int | str | float | bool | None]] = None,
    framework_version: str = "1.2-1",
    py_version: str = "py3",
) -> tuple[
    sagemaker.sklearn.SKLearn,
    sagemaker.sklearn.model.SKLearnPredictor,
    dict[str, str],
    dict[str, str],
]:
    """
    Deploy a Scikit-learn model using SageMaker. This function handles both training
    a new model in script mode and deploying an existing model.

    Args:
        name (str): The base name for the model and endpoint.
        role (str): The AWS IAM role for SageMaker permissions.
        image_uri (str): The URI of the Docker image to use for training/deployment.
        script_mode (bool): Whether to use script mode for training a new model.
        training_input_path (Optional[str]): S3 path to training data for script mode.
        model_data (Optional[str]): S3 path to the model artifact for existing model mode.
        entry_point (str): The Python script for training or inference.
        source_dir (str): The directory containing the entry point script.
        hyperparameters (Optional[dict]): Hyperparameters for the training job.
        framework_version (str): The version of Scikit-learn to use.
        py_version (str): The Python version to use.

    Returns:
        tuple: A tuple containing the model, predictor, model description, and endpoint description.

    Examples:
        >>> MODEL_PREFIX: str = "scikit-learn-script-mode-inference"
        >>> role: str = "arn:aws:iam::763678331342:role/temp_sagemaker_role"
        >>> MODEL_DATA: str = "s3://sagemaker-eu-central-1-763678331342/demo"
        >>> model, predictor, model_desc, endpoint_desc = deploy_sklearn_model(
        >>>     name=MODEL_PREFIX,
        >>>     model_data=model_data,
        >>>     role=role
        >>> )
    """
    # Set up logging
    logger.add(
        sink="train.log",
        format="{time:YYYY-MM-DD HH:MM:SS} | {level} | {message}",
        rotation="10 MB",
        retention="1 week",
        compression="zip",
    )

    # Input validation
    check_string_type(name)
    check_string_type(role)
    check_string_type(image_uri)
    check_string_type(entry_point)
    check_string_type(source_dir)
    check_string_type(framework_version)
    check_string_type(py_version)
    check_bool_type(script_mode)

    # Set model name with a timestamp
    model_name: str = f"{name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    if script_mode:
        logger.info("Script mode deployment initiated...")

        check_string_type(training_input_path)
        check_s3_path(training_input_path)

        # Create the SageMaker SKLearn model for training
        logger.info("Training SageMaker SKLearn model...")
        model: sagemaker.sklearn.SKLearn = SKLearn(
            base_job_name=model_name,
            name=model_name,
            entry_point=entry_point,
            role=role,
            instance_count=1,
            instance_type="ml.c5.xlarge",
            py_version=py_version,
            framework_version=framework_version,
            image_uri=image_uri,
            script_mode=True,
            source_dir=source_dir,
            hyperparameters=hyperparameters,
        )

        # Train the estimator
        logger.info("Training the estimator...")
        model.fit({"train": training_input_path})

    else:
        logger.info("Existing model deployment initiated...")

        check_s3_path(model_data)

        # Create the SageMaker SKLearn model for deployment
        logger.info("Creating SageMaker SKLearn model...")
        model: sagemaker.sklearn.model.SKLearnModel = SKLearnModel(
            name=model_name,
            model_data=model_data,
            role=role,
            entry_point=entry_point,
            py_version=py_version,
            framework_version=framework_version,
            image_uri=image_uri,
            source_dir=source_dir,
        )

    # Prepare model description
    model_desc: dict[str, str] = dict(
        model_name=model.name if hasattr(model, "name") else None,
        base_job_name=model.base_job_name if hasattr(model, "base_job_name") else None,
        current_job_name=(model._current_job_name if hasattr(model, "_current_job_name") else None),
        model_data=model.model_data,
        image_uri=model.image_uri,
        source_dir=model.source_dir,
        entry_point=model.entry_point,
        role=model.role,
        framework_version=model.framework_version,
        py_version=model.py_version,
        model_dependencies=model.dependencies,
    )

    # Create SageMaker Endpoint
    logger.info("Creating SageMaker Endpoint...")
    endpoint_name: str = f"{name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    # Deploy the Model as a SageMaker Endpoint
    predictor: sagemaker.sklearn.model.SKLearnPredictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.c5.large",
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
    )
    logger.info("Model deployed successfully!")

    # Describe the SageMaker Endpoint
    endpoint_context: sagemaker.lineage.context.EndpointContext = predictor.endpoint_context()
    RUN_TIME_URL_PREFIX: str = "https://runtime.sagemaker.eu-central-1.amazonaws.com/endpoints/"
    RUN_TIME_URL_SUFFIX: str = "/invocations"
    logger.info(f"Model runtime URL: {RUN_TIME_URL_PREFIX}{predictor.endpoint_name}{RUN_TIME_URL_SUFFIX}")

    # Prepare endpoint description
    endpoint_desc: dict[str, str] = dict(
        endpoint_name=predictor.endpoint_name,
        endpoint_creation_time=endpoint_context.creation_time.strftime("%Y-%m-%d-%H-%M-%S"),
        endpoint_context_name=endpoint_context.context_name,
        run_time_url=f"{RUN_TIME_URL_PREFIX}{predictor.endpoint_name}{RUN_TIME_URL_SUFFIX}",
    )

    return model, predictor, model_desc, endpoint_desc
