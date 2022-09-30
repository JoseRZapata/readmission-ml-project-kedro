"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import evaluate_model, model_evaluation_check


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=evaluate_model,
                inputs=["predictions_train",
                        "y_train_model_input",
                        'params:train'],
                outputs="score_train",
                name="train_model_evaluation"
            ),
            node(
                func=evaluate_model,
                inputs=["predictions_test",
                        "y_test",
                        'params:test'],
                outputs="score_test",
                name="test_model_evaluation"
            ),
            node(
                func=model_evaluation_check,
                inputs=["x_train_model_input",
                        "x_test_transformed",
                        "y_train_model_input",
                        "y_test",
                        "model_trained"],
                outputs=None,
                name="model_evaluation_check"
            )
        ]
    )
