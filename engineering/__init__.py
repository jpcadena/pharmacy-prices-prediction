"""
Engineering package initialization
"""
import logging
import pandas as pd
from engineering.transformation import remove_missing_values

logger: logging.Logger = logging.getLogger(__name__)


def transform_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Transform dataframe based on the requirements
    :param dataframe: Raw dataframe
    :type dataframe: pd.DataFrame
    :return: Transformed dataframe
    :rtype: pd.DataFrame
    """
    logger.info("Running transform_data()")
    dataframe = remove_missing_values(dataframe)
    return dataframe
