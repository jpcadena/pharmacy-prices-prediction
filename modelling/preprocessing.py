"""
Preprocessing section including: Formatting, Cleaning, Anonymization, Sampling
"""
import logging

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

from core.decorators import with_logging, benchmark

logger: logging.Logger = logging.getLogger(__name__)


def convert_string_to_bool(value: str = 'approved_for_hospital_use') -> bool:
    """
    Convert the column value to a boolean
    :param value: The column value to convert
    :type value: str
    :return: The boolean value
    :rtype: bool
    """
    return value.lower() == 'oui'


def convert_str_pct_to_float(value: str = 'reimbursement_rate') -> np.float16:
    """
    Convert the column value to a float16
    :param value: The column value to convert
    :type value: str
    :return: The floating value
    :rtype: np.float16
    """
    return np.float16(value.replace('%', ''))


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


def oversample_data(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Perform oversampling on the specified column
    :param dataframe: The dataframe to oversample
    :type dataframe: pd.DataFrame
    :param column_name: The name of the column to oversample
    :type column_name: str
    :return: The oversampled dataframe
    :rtype: pd.DataFrame
    """
    # Get the minority class and its count
    minority_class = dataframe[column_name].value_counts().idxmin()
    minority_class_count = dataframe[column_name].value_counts()[
        minority_class]

    # Create a list of dataframes, one for each class
    dfs: list[pd.DataFrame] = [
        dataframe[dataframe[column_name] == cls] for cls in
        dataframe[column_name].unique()]

    # Oversample the minority class
    oversampled_minority = resample(
        dfs[dataframe[column_name].unique().index(minority_class)],
        replace=True, n_samples=dataframe[column_name].value_counts().max(),
        random_state=42)

    # Append the oversampled minority class to the list of dataframes
    dfs.append(oversampled_minority)

    # Concatenate the dataframes and return the result
    return pd.concat(dfs)


# def undersample_data(
#         dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
#     """
#     Perform under sampling on the specified column
#     :param dataframe: The dataframe to under-sample
#     :type dataframe: pd.DataFrame
#     :param column_name: The name of the column to under-sample
#     :type column_name: str
#     :return: The under-sampled dataframe
#     :rtype: pd.DataFrame
#     """
#     majority_class: str = dataframe[column_name].value_counts().idxmax()
#     majority_class_count: int = dataframe[column_name].value_counts()[
#         majority_class]
#     dfs: list[pd.DataFrame] = [
#         dataframe[dataframe[column_name] == cls] for cls in
#         dataframe[column_name].unique()]
#     majority_class_index: int = list(dataframe[column_name].unique()).index(
#         majority_class)
#     undersampled_majority: pd.DataFrame = dfs[majority_class_index].sample(
#         n=majority_class_count, random_state=42)
#     dfs.pop(majority_class_index)
#     dfs.append(undersampled_majority)
#     return pd.concat(dfs)

def undersample_data(
        dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Perform under-sampling on the specified column using
     RandomUnderSampler from imblearn.
    :param dataframe: The dataframe to under-sample
    :type dataframe: pd.DataFrame
    :param column_name: The name of the column to under-sample
    :type column_name: str
    :return: The under-sampled dataframe
    :rtype: pd.DataFrame
    """
    # separate features and target
    X = dataframe.drop(columns=[column_name])
    y = dataframe[column_name]

    # encode target column as numeric
    label_encoder: LabelEncoder = LabelEncoder()
    y_numeric = label_encoder.fit_transform(y)
    print(y_numeric)

    # define the undersampling strategy
    undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    # fit and apply the undersampling strategy
    # TODO: Requires all numerical features
    X_undersampled, y_undersampled = undersampler.fit_resample(X, y_numeric)
    # decode target column back to original categorical values
    y_undersampled = label_encoder.inverse_transform(y_undersampled)

    # combine the undersampled features and target into a new DataFrame
    df_undersampled = pd.concat([X_undersampled, y_undersampled], axis=1)
    return df_undersampled

# Other techniques could be: Cost-Sensitive Learning and Anomaly Detection


def scale_data(x_train: np.ndarray, x_test: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Scale the features data
    :param x_train: The training data
    :type x_train: np.ndarray
    :param x_test: The test data
    :type x_test: np.ndarray
    :return: The scaled data
    :rtype:
    """
    scaler: StandardScaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled
