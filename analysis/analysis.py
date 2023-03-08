"""
First Analysis script
"""
import logging
import pandas as pd
from core.decorators import with_logging, benchmark

logger: logging.Logger = logging.getLogger(__name__)
pd.set_option('display.max_columns', 20)


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
    print(dataframe.info(memory_usage='deep'))
    print(dataframe.memory_usage(deep=True))
    print(dataframe.describe(include='all', datetime_is_numeric=True))
    non_numeric_df = dataframe.select_dtypes(exclude=[
        'uint8', 'uint16', 'uint32', 'uint64',
        'int8', 'int16', 'int32',
        'int64',
        'float16', 'float32', 'float64'])
    for column in non_numeric_df.columns:
        print(column)
        print(non_numeric_df[column].value_counts())
        print(non_numeric_df[column].unique())
        print(non_numeric_df[column].value_counts(normalize=True) * 100)

    # Imbalanced classes:
    # administrative_status: 99.88-0.12% undersampling
    # marketing_status: 84.8-14.6-0.6-0.01% oversampling
    # approved_for_hospital_use: 82-18% oversampling
    # route_of_administration: 81.13-2.63-2.62-2.07-1.89-...%
    # marketing_authorization_status: 95.39-3.67-0.88-0.07%
    # marketing_authorization_process: 65.51-13.08-12.35-7.82% oversampling
