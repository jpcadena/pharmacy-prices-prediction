"""
Evaluation script including improvement and tests.
"""
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from core.decorators import with_logging, benchmark

logger: logging.Logger = logging.getLogger(__name__)


@with_logging
@benchmark
def evaluate_model(y_pred: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluate a binary classification ml_model based on several metrics.
    :param y_pred: Predicted binary labels
    :type y_pred: np.ndarray
    :param y_test: True binary values'
    :type y_test: np.ndarray
    :return: None
    :rtype: NoneType
    """
    mae: float = mean_absolute_error(y_test, y_pred)
    mse: float = mean_squared_error(y_test, y_pred)
    r_mse: float = np.sqrt(mse)
    determination_coefficient: float = r2_score(y_test, y_pred)
    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', r_mse)
    print('R-squared (R2):', determination_coefficient)
    logger.info("MAE: %f", mae)
    logger.info("MSE: %f", mse)
    logger.info("RMSE: %f", r_mse)
    logger.info("R2: %f", determination_coefficient)
