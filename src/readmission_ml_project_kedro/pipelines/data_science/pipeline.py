"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import predict, train_model, test_transform

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["x_train_model_input",
                        "y_train_model_input",
                        'parameters'],
                outputs="model_trained",
                name="train_model"
            ),
            node(
                func=predict,
                inputs=["model_trained",
                        "x_train_model_input"],
                outputs="predictions_train",
                name="predict_train"
            ),
            node(
                func=test_transform,
                inputs=["x_test", "column_transformers_pipeline"],
                outputs="x_test_transformed",
                name="x_test_transform",

            ),
            node(
                func=predict,
                inputs=["model_trained",
                        "x_test_transformed"],
                outputs="predictions_test",
                name="predict_test"
            )
        ]
    )

