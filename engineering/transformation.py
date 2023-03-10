"""
Transformation script for Engineering module
"""
import logging
import re
from typing import Any
import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)
# TODO: Fix the Feature engineering


def remove_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove missing values from the dataframe
    :param dataframe: Dirty dataframe to remove missing values from
    :type dataframe: pd.DataFrame
    :return: Cleaned dataframe
    :rtype: pd.DataFrame
    """
    if (dataframe == 0).any().any():
        logger.warning("FOUND 0 VALUES")
        zero_values = (dataframe == 0).sum()
        print(zero_values[zero_values > 0])
        print(zero_values[zero_values > 0] / dataframe.shape[0] * 100)
        dataframe = dataframe.replace(0, np.nan)
        dataframe = dataframe.dropna()
        dataframe = dataframe.copy
    return dataframe


def convert_date_column(
        dataframe: pd.DataFrame, date_column: str = "Purchase Date",
        date_format: str = "%Y-%m-%d"
) -> pd.DataFrame:
    """
    Convert a date column into a standard date format
    :param dataframe: DataFrame to convert its column
    :type dataframe: pd.DataFrame
    :param date_column: Name of dataframe date column to convert
    :type date_column: str
    :param date_format: Date format to use
    :type date_format: str
    :return: Converted dataframe with standard date format
    :rtype: pd.DataFrame
    """
    dataframe.loc[:, date_column] = pd.to_datetime(
        dataframe[date_column], format=date_format).dt.normalize()
    return dataframe


def pascal_to_snake(column_name: str) -> str:
    """
    Convert Pascal Columns names to snake_case
    :param column_name: Name of column
    :type column_name: str
    :return: Converted column name
    :rtype: str
    """
    column_name = column_name.replace(" ", "")
    words: list = re.findall(r'[A-Z][^A-Z]*', column_name)
    snake_case: str = '_'.join([word.lower() for word in words])
    return snake_case


def convert_column_names(dataframe: pd.DataFrame):
    """
    Cleaning and apply pascal_to_snake function
    :param dataframe: Dataframe to apply the function
    :type dataframe: pd.DataFrame
    :return: New Dataframe with converted column names
    :rtype: pd.DataFrame
    """
    dataframe.columns = [pascal_to_snake(col) for col in dataframe.columns]
    return dataframe


def strip_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Function to strip column names and cells in string columns
    :param dataframe: Dataframe to apply the function
    :type dataframe: pd.DataFrame
    :return: New Dataframe without blank spaces
    :rtype: pd.DataFrame
    """
    dataframe.columns = [col.strip() for col in dataframe.columns]
    for column in dataframe.columns:
        if dataframe[column].dtype == object:
            dataframe[column] = dataframe[column].str.strip()
    return dataframe
