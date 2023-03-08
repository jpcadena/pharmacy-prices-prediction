"""
Main script
"""
import logging

import numpy as np
import pandas as pd
from numpy import float16

from analysis import numerical_eda, visualize_data
from core import logging_config
from core.config import CHUNK_SIZE
from engineering import transform_data
from engineering.extraction import extract_raw_data
from engineering.persistence_manager import PersistenceManager, DataType
from engineering.transformation import find_missing_values
from modelling.preprocessing import undersample_data

logging_config.setup_logging()
logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main function to execute
    :return: None
    :rtype: NoneType
    """
    logger.info("Running main method")
    d_types: dict = {
        'drug_id': str, 'description': str,
        'administrative_status': 'category', 'marketing_status': 'category',
        'dosage_form': str, 'route_of_administration': str,
        'marketing_authorization_status': 'category',
        'marketing_authorization_process': 'category',
        'pharmaceutical_companies': str, 'price': float16}
    drugs_train: pd.Dataframe = extract_raw_data(
        filename='drugs_train.csv', d_types=d_types,
        parse_dates=['marketing_declaration_date',
                     'marketing_authorization_date'])
    # drugs_train_undersampled: pd.DataFrame = undersample_data(
    #     drugs_train, 'administrative_status')

    active_ingredients: pd.DataFrame = extract_raw_data(
        filename='active_ingredients.csv', parse_dates=None)
    drug_label_feature_eng: pd.DataFrame = extract_raw_data(
        filename='drug_label_feature_eng.csv', parse_dates=None)

    dataframe: pd.DataFrame = drugs_train.merge(
        active_ingredients, on="drug_id")
    dataframe = dataframe.merge(drug_label_feature_eng, on="description")

    dataframe = dataframe.dropna(axis=1)
    dataframe = dataframe.drop(["drug_id", "description"], axis=1)
    print(dataframe.shape)
    print(dataframe.dtypes)

    dataframe = pd.get_dummies(
        dataframe, columns=[
            "dosage_form", "route_of_administration",
            "active_ingredient", "pharmaceutical_companies"])
    print(dataframe.shape)
    print(dataframe.dtypes)

    # Fixme: Undersample with numerical values
    dataframe_undersampled: pd.DataFrame = undersample_data(
        dataframe, 'administrative_status')
    print(dataframe_undersampled.head)
    print(dataframe_undersampled.shape)
    print(dataframe.dtypes)

    # dataframe["num_dosage_forms"] = dataframe["description"].apply(
    #     lambda x: int(x.split()[0]) if x.split()[0].isdigit() else 0)
    # dataframe["marketing_declaration_year"] = dataframe[
    #     "marketing_declaration_date"].apply(
    #     lambda x: int(str(x)[:4]))
    # dataframe["marketing_authorization_year"] = dataframe[
    #     "marketing_authorization_date"].apply(lambda x: int(str(x)[:4]))
    # count_cols = [col for col in dataframe.columns if
    #               col.startswith("count_")]
    # for col in count_cols:
    #     dataframe[col.replace("count", "has")] = np.where(
    #         dataframe[col] > 0, 1, 0)
    # try:
    #     numerical_eda(dataframe)
    # except Exception as exc:
    #     logger.error(exc)
    # # visualize_data(dataframe)
    # PersistenceManager.save_to_pickle(dataframe)
    # # dataframe = transform_data(dataframe)
    # print(dataframe)
    # saved: bool = PersistenceManager.save_to_csv(dataframe)
    # print(saved)


if __name__ == '__main__':
    logger.info("First log message")
    main()
    logger.info("End of the program execution")
