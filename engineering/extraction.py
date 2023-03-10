"""
Extraction script for Engineering module
"""
import pandas as pd
from numpy import float16
from core.config import CHUNK_SIZE
from engineering.persistence_manager import PersistenceManager, DataType


def extract_main_data(
        filename: str = 'drugs_train.csv', data_type: DataType = DataType.RAW,
        chunk_size: int = CHUNK_SIZE, train: bool = False
) -> pd.DataFrame:
    """
    Engineering method to extract the main raw data from csv file
    :param filename: Filename to extract data from. The default is
     'drugs_train.csv'
    :type filename: str
    :param data_type: Path where data will be saved: RAW or
     PROCESSED. The default is RAW
    :type data_type: DataType
    :param chunk_size: Number of chunks to split dataset. The default
     is CHUNK_SIZE
    :type chunk_size: int
    :param train: True if the dataframe will be for training; False
     otherwise
    :type train: bool
    :return: Dataframe with raw data
    :rtype: pd.DataFrame
    """
    d_types: dict = {
        'drug_id': str, 'description': str,
        'administrative_status': 'category', 'marketing_status': 'category',
        'dosage_form': str, 'route_of_administration': str,
        'marketing_authorization_status': 'category',
        'marketing_authorization_process': 'category',
        'pharmaceutical_companies': str}
    if train:
        d_types['price'] = float16
    parse_dates: list[str] = [
        'marketing_declaration_date', 'marketing_authorization_date']
    dataframe: pd.DataFrame = PersistenceManager.load_from_csv(
        filename=filename, data_type=data_type, chunk_size=chunk_size,
        dtypes=d_types, parse_dates=parse_dates)
    return dataframe


def extract_raw_data(
        filename: str = 'active_ingredients.csv',
        data_type: DataType = DataType.RAW, chunk_size: int = CHUNK_SIZE
) -> pd.DataFrame:
    """
    Engineering method to extract raw data from csv file
    :param filename: Filename to extract data from. The default is
     'active_ingredients.csv'
    :type filename: str
    :param data_type: Path where data will be saved: RAW or
     PROCESSED. The default is RAW
    :type data_type: DataType
    :param chunk_size: Number of chunks to split dataset. The default
     is CHUNK_SIZE
    :type chunk_size: int
    :return: Dataframe with raw data
    :rtype: pd.DataFrame
    """
    dataframe: pd.DataFrame = PersistenceManager.load_from_csv(
        filename=filename, data_type=data_type, chunk_size=chunk_size)
    return dataframe
