"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.18.2
"""
import logging

import numpy as np
import pandas as pd
from typing import Any, Dict
from sklearn.metrics import recall_score
import mlflow

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation

logger = logging.getLogger(__name__)

def evaluate_model(predictions: np.ndarray,
                   test_labels: pd.Series,
                   name: str):
    """
    Evaluate the model by calculating the accuracy score.
    """
    score = recall_score(test_labels, predictions)

    logger.info(f"Model accuracy {name}= {score}")

    mlflow.set_experiment('readmission')
    mlflow.log_metric(f"recall {name}", score)

    # parse the score to a string with only 4 decimal places
    return f"{score:.4f}"


def model_evaluation_check(x_train: np.ndarray,
                           x_test: np.ndarray,
                           y_train: np.ndarray,
                           y_test: pd.Series,
                           model):

    y_test = y_test['readmitted'].to_numpy()
    train_ds = Dataset(x_train, label=y_train, cat_features=[])
    test_ds = Dataset(x_test, label=y_test, cat_features=[])

    evaluation_suite = model_evaluation()
    suite_result = evaluation_suite.run(train_ds, test_ds, model)
    mlflow.log_param(f"model evaluation validation", str(suite_result.passed()))
    if not suite_result.passed():
        #save report in data/08_reporting
        suite_result.save_as_html('data/08_reporting/model_eval_check.html')
        logger.error("model not pass evaluation tests")
        print("model not pass evaluation tests")

