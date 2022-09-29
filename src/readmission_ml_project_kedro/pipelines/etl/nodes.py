"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.2
"""
import importlib
import logging
from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
import numpy as np

# Assemble pipeline(s)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
from deepchecks.tabular.suites import train_test_validation

import great_expectations as ge
from great_expectations.checkpoint.types.checkpoint_result import CheckpointResult

logger = logging.getLogger(__name__)


def get_data(parameters: Dict[str, Any]) -> pd.DataFrame:
    """Downloads data from url.
    Args:
        parameters['url']: Url to download data from.
    Returns: dataframe containing data.
    """
    return pd.read_csv(parameters['url'])


def etl_processing(data: pd.DataFrame,
                   parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    General transformations to the data like removing columns with
    the same constant value, duplicated columns., duplicate values

    Args:
        data: raw data after extract
        parameters: list of the general transforms to apply to all the data

    Returns:
        pd.DataFrame: transformed data

    """
    mlflow.set_experiment('readmission')
    mlflow.log_param("shape raw_data", data.shape)

    data = (data
            .replace('?', np.nan)
            .pipe(sort_data, col = 'readmitted')
            .pipe(drop_duplicates, drop_cols=['patient_nbr'])
    )

    # convert this step as a scikit-learn transformer
    # this steps is only useful until at the end in the SLQ query or
    # data load method, only query the specific columns
    # the specific columns
    columns = parameters['features']
    # convert target column (str) to list
    target = parameters['target_column']
    columns.append(target)

    pipe_functions = []
    methods = []

    for x in parameters['data_train_transforms']:
        lib: str = x['lib']
        name: str = x['name']
        method: str = x['method']
        params: str = x['params']
        methods.append((method, params))

        class_lib = importlib.import_module(lib)
        pipe_functions.append((name, getattr(class_lib, method)(**params)))

    mlflow.log_param('etl_transforms', methods)

    pipeline_train_data = Pipeline(steps=pipe_functions)
    data_transformed = pipeline_train_data.fit_transform(data)

    data_transformed = (data_transformed
                        .pipe(sort_data, col = 'readmitted')
                        .pipe(drop_duplicates, drop_cols=['patient_nbr'])
    )
    mlflow.log_param("shape data etl", data_transformed.shape)
    return data_transformed


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.
    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_processing.yml.
    Returns:
        Split data.
    """
    mlflow.set_experiment('readmission')
    mlflow.log_param("split random_state", parameters['split']['random_state'])
    mlflow.log_param("split test_size", parameters['split']['test_size'])

    #remove rows without target information
    data = data.dropna(subset=[parameters['target_column']])

    x_features = data[parameters['features']]
    y_target = data[parameters['target_column']]

    x_train, x_test, y_train, y_test = train_test_split(
        x_features,
        y_target,
        test_size=parameters['split']['test_size'],
        random_state=parameters['split']['random_state']
    )

    mlflow.log_param(f"shape train", x_train.shape)
    mlflow.log_param(f"shape test", x_test.shape)

    return x_train, x_test, y_train, y_test


def data_integrity_validation(data: pd.DataFrame,
                              parameters: Dict) -> pd.DataFrame:

    categorical_features = parameters['categorical_cols']
    label = parameters['target_column']

    dataset = Dataset(data,
                 label=label,
                 cat_features=categorical_features)

    # Run Suite:
    integ_suite = data_integrity()
    suite_result = integ_suite.run(dataset)
    mlflow.log_param(f"data integrity validation", str(suite_result.passed()))
    if not suite_result.passed():
        # save report in data/08_reporting
        suite_result.save_as_html('data/08_reporting/data_integrity_check.html')
        logger.error("data integrity not pass validation tests")
        #raise Exception("data integrity not pass validation tests")
    return data


def validation_data(data: pd.DataFrame) -> pd.DataFrame:
    context = ge.get_context()
    result: CheckpointResult = context.run_checkpoint(checkpoint_name="check_datos",
                                     batch_request = None,
                                     run_name = None
                                     )
    if not result["success"]:
        logger.error("Dataset not pass data validation tests")
        raise Exception("Dataset not pass data validation tests")
    return data


def train_test_validation_dataset(x_train,
                                  x_test,
                                  y_train,
                                  y_test,
                                  parameters: Dict) -> Tuple:
    categorical_features = parameters['categorical_cols']
    label = parameters['target_column']

    train_df = pd.concat([x_train, y_train], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)

    train_ds = Dataset(train_df,
                       label=label,
                       cat_features=categorical_features
                       )
    test_ds = Dataset(test_df,
                      label=label,
                      cat_features=categorical_features
                      )
    validation_suite = train_test_validation()
    suite_result = validation_suite.run(train_ds, test_ds)
    mlflow.log_param("train_test validation", str(suite_result.passed()))
    if not suite_result.passed():
        # save report in data/08_reporting
        suite_result.save_as_html('data/08_reporting/train_test_check.html')
        logger.error("Train / Test Dataset not pass validation tests")
    return x_train, x_test, y_train, y_test

# --- help functions -------
def sort_data(data: pd.DataFrame, col: str) -> pd.DataFrame:
    "Sort data by and specific column"
    data = data.sort_values(by=col,ascending= False)
    return data

# remove duplicates from data based on a column
def drop_duplicates(data: pd.DataFrame,
                    drop_cols: list) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    data = data.drop_duplicates(subset=drop_cols, keep='first')    
    return data