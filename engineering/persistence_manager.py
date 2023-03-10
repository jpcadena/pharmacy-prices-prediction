"""
Persistence script
"""
import logging
from enum import Enum
from typing import Optional

import joblib
import pandas as pd
from catboost import CatBoostRegressor
from keras import Sequential
from keras.models import save_model as save_nn_model
from lightgbm import LGBMRegressor
from numpy import float16
from pandas.io.parsers import TextFileReader
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from core.config import ENCODING
from core.decorators import with_logging, benchmark
from modelling.preprocessing import convert_string_to_bool, \
    convert_str_pct_to_float

logger: logging.Logger = logging.getLogger(__name__)


class DataType(Enum):
    """
    Data Type class based on Enum
    """
    RAW: str = 'data/raw/'
    PROCESSED: str = 'data/processed/'
    FIGURES: str = 'reports/figures/'
    MODELS: str = 'models/'


class PersistenceManager:
    """
    Persistence Manager class
    """

    @staticmethod
    @with_logging
    @benchmark
    def save_to_csv(
            dataframe: pd.DataFrame,
            data_type: DataType = DataType.PROCESSED,
            filename: str = 'data.csv') -> bool:
        """
        Save a dataframe as csv file
        :param dataframe: The dataframe to save
        :type dataframe: pd.DataFrame
        :param data_type: folder where data will be saved: RAW or
         PROCESSED
        :type data_type: DataType
        :param filename: name of the file
        :type filename: str
        :return: confirmation for csv file created
        :rtype: bool
        """
        filepath: str = f'{str(data_type.value)}{filename}'
        dataframe.to_csv(filepath, index=False, encoding=ENCODING)
        logger.info('Dataframe saved as csv: %s', filepath)
        return True

    @staticmethod
    @with_logging
    @benchmark
    def load_from_csv(
            filename: str, data_type: DataType, chunk_size: int,
            dtypes: Optional[dict] = None,
            parse_dates: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Load dataframe from CSV using chunk scheme
        :param filename: name of the file
        :type filename: str
        :param data_type: Path where data will be saved: RAW or
         PROCESSED
        :type data_type: DataType
        :param chunk_size: Number of chunks to split dataset
        :type chunk_size: int
        :param dtypes: Dictionary of columns and datatypes
        :type dtypes: dict
        :param parse_dates: List of date columns to parse
        :type parse_dates: list[str]
        :return: dataframe retrieved from CSV after optimization with
         chunks
        :rtype: pd.DataFrame
        """

        converters: dict = {
            'approved_for_hospital_use': convert_string_to_bool,
            'reimbursement_rate': convert_str_pct_to_float
        }

        filepath: str = f'{data_type.value}{filename}'
        text_file_reader: TextFileReader = pd.read_csv(
            filepath, header=0, chunksize=chunk_size, encoding=ENCODING,
            converters=converters, parse_dates=parse_dates)
        dataframe: pd.DataFrame = pd.concat(
            text_file_reader, ignore_index=True)
        if dtypes:
            for key, value in dtypes.items():
                if value == float16:
                    try:
                        dataframe[key] = pd.to_numeric(dataframe[key],
                                                       errors='coerce')
                        dataframe[key] = dataframe[key].astype(value)
                    except Exception as exc:
                        logger.error(exc)
                else:
                    try:
                        dataframe[key] = dataframe[key].astype(value)
                    except Exception as exc:
                        logger.error(exc)

        logger.info('Dataframe loaded from csv: %s', filepath)
        return dataframe

    @staticmethod
    @with_logging
    @benchmark
    def save_to_pickle(
            dataframe: pd.DataFrame, filename: str = 'optimized_df.pkl',
            data_type: DataType = DataType.PROCESSED
    ) -> None:
        """
        Save dataframe to pickle file
        :param dataframe: dataframe
        :type dataframe: pd.DataFrame
        :param filename: name of the file
        :type filename: str
        :param data_type: Path where data will be saved: RAW or PROCESSED
        :type data_type: DataType
        :return: None
        :rtype: NoneType
        """
        filepath: str = f'{data_type.value}{filename}'
        dataframe.to_pickle(filepath)
        logger.info('Dataframe saved as pickle: %s', filepath)

    @staticmethod
    @with_logging
    @benchmark
    def load_from_pickle(
            filename: str = 'optimized_df.pkl',
            data_type: DataType = DataType.PROCESSED
    ) -> pd.DataFrame:
        """
        Load dataframe from Pickle file
        :param filename: name of the file to search and load
        :type filename: str
        :param data_type: Path where data will be loaded from: RAW or
         PROCESSED
        :type data_type: DataType
        :return: dataframe read from pickle
        :rtype: pd.DataFrame
        """
        filepath: str = f'{data_type.value}{filename}'
        dataframe: pd.DataFrame = pd.read_pickle(filepath)
        logger.info('Dataframe loaded from pickle: %s', filepath)
        return dataframe


def save_model(model, model_name: str, data_type: DataType = DataType.MODELS):
    """
    Save a machine learning model to a file using joblib or Keras.
    :param model: The model instance
    :type model: obj
    :param model_name: The model name
    :type model_name: str
    :param data_type: Path where the model will be saved. The default
     is MODELS
    :type data_type: DataType
    :return: None
    :rtype: NoneType
    """
    # Fixme: Refactor function to not use if-clauses and isinstance
    if isinstance(model, XGBRegressor):
        file_extension = '.json'
        model.save_model(
            f'{data_type.value}{model_name}{file_extension}')
    elif isinstance(model, CatBoostRegressor):
        file_extension = '.cbm'
        model.save_model(f'{data_type.value}{model_name}{file_extension}')
    elif isinstance(model, LGBMRegressor):
        file_extension = '.txt'
        model.booster_.save_model(
            f'{data_type.value}{model_name}{file_extension}')
    elif isinstance(model, (
            AdaBoostRegressor, DecisionTreeRegressor, LinearRegression,
            GradientBoostingRegressor, KNeighborsRegressor, SGDRegressor)):
        file_extension = '.joblib'
        joblib.dump(model, f'{data_type.value}{model_name}{file_extension}')
    elif isinstance(model, Sequential):
        file_extension = '.h5'
        save_nn_model(model, f'{data_type.value}{model_name}{file_extension}')
    else:
        raise ValueError(f"Model type not supported: {type(model)}")
