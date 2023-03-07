"""
Preprocessing section including: Formatting, Cleaning, Anonymization, Sampling
"""
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from core.decorators import with_logging, benchmark

logger: logging.Logger = logging.getLogger(__name__)


@with_logging
@benchmark
def downcast_type(dataframe: pd.DataFrame):
    """
    Optimization of numeric columns by down-casting its datatype
    :param dataframe: dataframe to optimize
    :type dataframe: pd.DataFrame
    :return: optimized dataframe
    :rtype: pd.DataFrame
    """
    numerics: list[str] = [
        'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32',
        'int64']
    numeric_ranges: list[tuple] = [
        (0, 255), (0, 65535), (0, 4294967295), (0, 18446744073709551615),
        (-128, 127), (-32768, 32767), (-2147483648, 2147483647),
        (-18446744073709551616, 18446744073709551615)]
    df_num_cols: pd.DataFrame = dataframe.select_dtypes(include=numerics)
    for column in df_num_cols:
        new_type: str = numerics[numeric_ranges.index(
            [num_range for num_range in numeric_ranges if
             df_num_cols[column].min() > num_range[0] and
             num_range[1] <= df_num_cols[column].max()][0])]
        df_num_cols[column] = df_num_cols[column].apply(
            pd.to_numeric, downcast=new_type)  # check map for Pd.Series
    dataframe[df_num_cols.columns] = df_num_cols
    return dataframe


@with_logging
@benchmark
def lof_observation(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function identifies outliers with LOF method
    :param dataframe: Dataframe containing data
    :type dataframe: pd.DataFrame
    :return: clean dataframe without outliers from LOF
    :rtype: pd.DataFrame
    """
    numerics: list[str] = [
        'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_num_cols: pd.DataFrame = dataframe.select_dtypes(include=numerics)
    df_outlier: pd.DataFrame = df_num_cols.astype("float64")
    clf: LocalOutlierFactor = LocalOutlierFactor(
        n_neighbors=20, contamination=0.1)
    clf.fit_predict(df_outlier)
    df_scores = clf.negative_outlier_factor_
    scores_df: pd.DataFrame = pd.DataFrame(np.sort(df_scores))
    scores_df.plot(stacked=True, xlim=[0, 20], color='r',
                   title='Visualization of outliers according to the LOF '
                         'method', style='.-')
    plt.savefig('reports/figures/outliers.png')
    plt.show()
    th_val = np.sort(df_scores)[2]
    outliers: bool = df_scores > th_val
    dataframe: pd.DataFrame = dataframe.drop(df_outlier[~outliers].index)
    logger.info("Dataframe shape: %s", dataframe.shape)
    return dataframe


@with_logging
@benchmark
def clear_outliers(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This function remove the outliers from specific column
    :param dataframe: Dataframe containing data
    :type dataframe: pd.DataFrame
    :param column: Column name
    :type column: str
    :return: clean dataframe from outliers using IQR
    :rtype: pd.DataFrame
    """
    first_quartile: float = dataframe[column].quantile(0.25)
    third_quartile: float = dataframe[column].quantile(0.75)
    iqr: float = third_quartile - first_quartile
    lower: float = first_quartile - 1.5 * iqr
    upper: float = third_quartile + 1.5 * iqr
    print(f"{column}- Lower score: ", lower, "and upper score: ", upper)
    logger.info(
        "%s - Lower score: %s and Upper score: %s", column, lower, upper)
    df_outlier = dataframe[column][(dataframe[column] > upper)]
    print(df_outlier)
    logger.warning(df_outlier.shape)
    return dataframe
