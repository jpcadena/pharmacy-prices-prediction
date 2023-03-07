"""
First Analysis script
"""
import logging
import pandas as pd
from core.decorators import with_logging, benchmark

logger: logging.Logger = logging.getLogger(__name__)


@with_logging
@benchmark
def analyze_dataframe(dataframe: pd.DataFrame) -> None:
    """
    Analyze the dataframe and its columns with inference statistics
    :param dataframe: DataFrame to analyze
    :type dataframe: pd.DataFrame
    :return: None
    :rtype: NoneType
    """
    print(dataframe.head())
    print(dataframe.shape)
    logger.info(dataframe.shape)
    print(dataframe.dtypes)
    logger.info(dataframe.dtypes)
    print(dataframe.info(memory_usage='deep'))
    print(dataframe.memory_usage(deep=True))
    print(dataframe.describe(include='all', datetime_is_numeric=True))
    non_numeric_df = dataframe.select_dtypes(exclude=[
        'uint8', 'uint16', 'uint32', 'uint64',
        'int8', 'int16', 'int32',
        'int64',
        'float16', 'float32', 'float64'])
    for column in non_numeric_df.columns:
        print(non_numeric_df[column].value_counts())
        print(non_numeric_df[column].unique())
        print(non_numeric_df[column].value_counts(normalize=True) * 100)
