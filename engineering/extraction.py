"""
Extraction script for Engineering module
"""
from typing import Optional
import pandas as pd
from core.config import CHUNK_SIZE
from engineering.persistence_manager import PersistenceManager, DataType


def extract_raw_data(
        filename: str = 'drugs_train.csv', data_type: DataType = DataType.RAW,
        chunk_size: int = CHUNK_SIZE, d_types: Optional[dict] = None,
        parse_dates: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Engineering method to extract raw data from csv file
    :param filename: Filename to extract data from. The default is
     'drugs_train.csv'
    :type filename: str
    :param data_type: Path where data will be saved: RAW or
     PROCESSED. The default is RAW
    :type data_type: DataType
    :param chunk_size: Number of chunks to split dataset. The default
     is CHUNK_SIZE
    :type chunk_size: int
    :param d_types: Optional dictionary to handle data types of columns.
     The default is None
    :type d_types: dict
    :param parse_dates: List of date columns to parse. The default is
     None
    :type parse_dates: list[str]
    :return: Dataframe with raw data
    :rtype: pd.DataFrame
    """
    dataframe: pd.DataFrame = PersistenceManager.load_from_csv(
        filename=filename, data_type=data_type, chunk_size=chunk_size,
        dtypes=d_types,
        parse_dates=parse_dates)
    return dataframe
