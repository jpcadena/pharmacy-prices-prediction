"""
Extraction script for Engineering module
"""
import pandas as pd
from numpy import float16
from core.config import CHUNK_SIZE
from engineering.persistence_manager import PersistenceManager, DataType


def extract_raw_data(
        filename: str = 'drugs_train.csv', data_type: DataType = DataType.RAW,
        chunk_size: int = CHUNK_SIZE, parse_dates: list[str] | None = None
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
    :param parse_dates: List of date columns to parse. The default is
     None
    :type parse_dates: list[str]
    :return: Dataframe with raw data
    :rtype: pd.DataFrame
    """
    # approved_for_hospital_use: bool (oui=True, non=False),
    # reimbursement_rate: float (%)
    d_types: dict = {
        'drug_id': str, 'description': str,
        'administrative_status': 'category', 'marketing_status': 'category',
        'dosage_form': str, 'route_of_administration': str,
        'marketing_authorization_status': 'category',
        'marketing_declaration_date': str, 'marketing_authorization_date': str,
        'marketing_authorization_process': 'category',
        'pharmaceutical_companies': str, 'price': float16}
    if not parse_dates:
        parse_dates = ['marketing_declaration_date',
                       'marketing_authorization_date']
    dataframe: pd.DataFrame = PersistenceManager.load_from_csv(
        filename=filename, data_type=data_type, chunk_size=chunk_size,
        dtypes=d_types,
        parse_dates=parse_dates)
    print(dataframe.dtypes)
    return dataframe
