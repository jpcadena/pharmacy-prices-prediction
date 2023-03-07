"""
This script contains a function to split the data into training and
 testing sets for a machine learning model.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from core.decorators import with_logging, benchmark

logger: logging.Logger = logging.getLogger(__name__)


@with_logging
@benchmark
def training(
        x_matrix: np.ndarray, dataframe: pd.DataFrame,
        target_column: str = 'insecurity'
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into training and testing sets for a machine
     learning model. The function takes a bag-of-words representation
      of the text data and a Pandas DataFrame containing the target
       variable. The target variable is specified by the user as a
        column name in the DataFrame. The function uses the
         train_test_split function from scikit-learn to split the data
          into training and testing sets with a test size of 20%.
           The function returns the training and testing sets for both
            the input features and the target variable as NumPy arrays
    :param x_matrix: A NumPy array with the data
    :type x_matrix: np.ndarray
    :param dataframe: A Pandas DataFrame containing the target variable
    :type dataframe: pd.DataFrame
    :param target_column: The name of the target variable column in the
     DataFrame
    :type target_column: str
    :return: A tuple containing the training and testing sets for both
     the input features and the target variable
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    x_array: np.ndarray = x_matrix
    y_array: np.ndarray = dataframe[target_column].values
    x_train, x_test, y_train, y_test = train_test_split(
        x_array, y_array, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test
