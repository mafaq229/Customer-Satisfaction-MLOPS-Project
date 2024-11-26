import logging

import pandas as pd
from zenml import step

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelNameConfig = ModelNameConfig()
    ) -> RegressorMixin:
    """
    Trains the model on the the ingested data
    Args:
        X_train: Training data
        y_train: trainig labels
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported.")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
