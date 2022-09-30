"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.2
"""
import importlib
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB


logger = logging.getLogger(__name__)


def train_model(x_features: np.ndarray,
                y_target: np.ndarray,
                parameters: Dict[str, Any]):
    """Trains the classification model.
    Args:
        x_features: Training data
        y_target: target column name
        parameters: model parameters
    Returns:
        Trained model.
    """
    x_train = x_features
    y_train = y_target

    logger.info("training model")
    
    model = GaussianNB()

    mlflow.set_experiment('readmission')
    mlflow.set_tag("mlflow.runName", model.__class__.__name__)

    model.fit(x_train, y_train)
    return model


def test_transform(x_test: pd.DataFrame,
                   train_transformer: Pipeline) -> np.ndarray:
    """ predictions for dataframe"""
    logger.info("transform X_test")
    x_test_transformed = train_transformer.transform(x_test)
    mlflow.set_experiment('readmission')
    mlflow.log_param(f"shape test_transformed", x_test_transformed.shape)
    return x_test_transformed


def predict(model,
            data: np.ndarray) -> np.ndarray:
    """ predictions for dataframe"""
    logger.info("predicting ")

    pred = model.predict(data)
    return pred
