"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (pre_processing, 
                    first_processing,
                    numerical_pipeline,
                    categorical_pipeline,
                    data_type_split,
                    last_processing, post_processing,
                    )


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [   node(
                func=pre_processing,
                inputs=["x_train","y_train", "parameters"],
                outputs=["x_train_out", "y_train_out"],
                name="pre_processing",
            ),
            node(
                func=first_processing,
                inputs=["x_train_out", "parameters"],
                outputs=["data_first", "first_processing_pipline"],
                name="first_processing",
            ),
            node(
                func=data_type_split,
                inputs=["data_first", "parameters"],
                outputs=["numerical_cols", "categorical_cols"],
                name="data_type_split",
            ),
            node(
                func=numerical_pipeline,
                inputs=["numerical_cols", "parameters"],
                outputs="numerical_pipeline",
                name="numerical_pipeline_transforms",
            ),
            node(
                func=categorical_pipeline,
                inputs=["categorical_cols", "parameters"],
                outputs="categorical_pipeline",
                name="categorical_pipeline_transforms",
            ),

            node(
                func=last_processing,
                inputs=["x_train_out",
                        "first_processing_pipline",
                        "numerical_pipeline",
                        "categorical_pipeline"],
                outputs=["column_transformers_pipeline", "x_train_transformed"],
                name="cols_transforms_pipeline",
            ),
            node(
                func=post_processing,
                inputs=["x_train_transformed",
                        "y_train_out"],
                outputs=["x_train_model_input",
                         "y_train_model_input"],
                name="post_processing",
            )

        ]
    )
