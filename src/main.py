from pprint import pprint

import boto3
import sagemaker

from src.code.utils import delete_s3_prefix
from src.script.deploy import deploy_sklearn_model

# Setup
PROFILE_NAME: str = "763678331342_ProgrammaticAccessUser"
boto_session: boto3.session.Session = boto3.Session(profile_name=PROFILE_NAME)
sagemaker_session: sagemaker.Session = sagemaker.Session(boto_session=boto_session)
REGION: str = "eu-central-1"
bucket: str = sagemaker_session.default_bucket()
ROLE_PREFIX: str = "arn:aws:iam::763678331342:role/"
ROLE_NAME: str = "temp_sagemaker_role"
role: str = f"{ROLE_PREFIX}{ROLE_NAME}"

FRAMEWORK: str = "sklearn"
IMAGE_SCOPE: str = "inference"
FRAMEWORK_VERSION: str = "1.2-1"
PY_VERSION: str = "py3"

# Create a SageMaker Model
image_uri: str = sagemaker.image_uris.retrieve(
    framework=FRAMEWORK,
    region=REGION,
    version=FRAMEWORK_VERSION,
    image_scope=IMAGE_SCOPE,
)

# Input data
# Upload data to default S3 bucket
# LOCAL_DATA_PATH: str = "data/data.parquet"
# import pandas as pd
# pd.read_parquet(DATA_PATH)
DATA_PREFIX: str = "script_mode"  # _TRAIN_DATA

# training_input_path: str = sagemaker_session.upload_data(
#     path=LOCAL_DATA_PATH,
#     bucket=bucket,
#     key_prefix=f"{DATA_PREFIX}/training",
# )
# "s3://sagemaker-eu-central-1-763678331342/script_mode/training/data.parquet"
training_input_path: str = f"s3://{bucket}/{DATA_PREFIX}/training/data.parquet"
model_data: str = f"s3://{bucket}/scikit-learn/model/model.tar.gz"
# Create the SageMaker SKLearn model
MODEL_PREFIX: str = "scikit-learn-script-mode-inference"

PARAMS: dict[str, int] = dict(n_estimators=20, random_state=123)

model, predictor, model_desc, endpoint_desc = deploy_sklearn_model(
    name=MODEL_PREFIX,
    # training_input_path=training_input_path,
    model_data=model_data,
    role=role,
    image_uri=image_uri,
    script_mode=False,
    # hyperparameters=dict(n_estimators=20),
)

# Show the model and endpoint descriptions
pprint(model_desc)
pprint(endpoint_desc)

# Delete the SageMaker Endpoint
predictor.delete_model()
predictor.delete_endpoint()
delete_s3_prefix(
    s3_data_path=f"s3://sagemaker-eu-central-1-763678331342/{model_desc.get('model_name')}"  # base_job_name
)
