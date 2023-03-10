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


def cast_column(
        dataframe: pd.DataFrame, column: str = 'Buyer Gender',
        d_type: Any = 'category') -> pd.DataFrame:
    """
    Simple transformation of datatype for a column into category
    :param dataframe: Dataframe to transform
    :type dataframe: pd.DataFrame
    :param column: Dataframe column to cast
    :type column: str
    :param d_type: Data type to cast to
    :type d_type: Any
    :return: Converted dataframe
    :rtype: pd.DataFrame
    """
    dataframe.loc[:, column] = dataframe[column].astype(d_type)
    return dataframe


def find_missing_values(
        dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove missing values from the dataframe
    :param dataframe: Dirty dataframe to remove missing values from
    :type dataframe: pd.DataFrame
    :return: Cleaned dataframe
    :rtype: pd.DataFrame
    """
    missing_values: pd.Series = (dataframe.isnull().sum())
    print(missing_values)
    if len(missing_values) > 0:
        logger.warning("FOUND MISSING VALUES")
        print("FOUND MISSING VALUES")
        print(missing_values[missing_values > 0])
        print(missing_values[missing_values > 0] / dataframe.shape[0] * 100)
        dataframe = dataframe.copy()
    return dataframe


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


def create_sale_year(
        dataframe: pd.DataFrame, new_column: str = 'Purchase Year',
        date_column: str = 'Purchase Date') -> pd.DataFrame:
    """
    Create a new column to store the year of the sale
    :param dataframe: Dataframe to manipulate
    :type dataframe: pd.DataFrame
    :param new_column: New column for year of the sale
    :type new_column: str
    :param date_column: Column of the sale date
    :type date_column: str
    :return: Updated Dataframe containing the year of the sale
    :rtype: pd.DataFrame
    """
    dataframe.loc[:, new_column] = dataframe[date_column].dt.year
    # dataframe[new_column] = dataframe[new_column].astype(uint16)
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
