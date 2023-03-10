"""
Neural Network using TensorFlow script
"""
import logging
import numpy as np
import pandas as pd

from modelling.preprocessing import scale_data
from modelling.train import training
from core.decorators import with_logging, benchmark

logger: logging.Logger = logging.getLogger(__name__)


@with_logging
@benchmark
def predict_model(
        dataframe: pd.DataFrame, ml_model,
        target_column: str = 'price', boost: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predicts the target variable values using the provided model and
     returns the predicted and actual values.
    :param dataframe: The pandas dataframe containing the data and the
    target variable
    :type dataframe: pd.DataFrame
    :param ml_model: The machine learning model to use for prediction
    :type ml_model: Any
    :param target_column: The name of the target variable column in the
     dataframe
    :type target_column: str
    :param boost: Whether to boost the model training by converting
     data to float32
    :type boost: bool
    :return: A tuple of predicted and actual values for the target
     variable
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    x_train, x_test, y_train, y_test = training(dataframe, target_column)
    if boost:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')
    x_train, x_test = scale_data(x_train, x_test)
    ml_model.fit(x_train, y_train)
    y_pred: np.ndarray = ml_model.predict(x_test)
    logger.info("y_pred: %s", y_pred)
    logger.info("y_test: %s", y_test)
    return y_pred, y_test
