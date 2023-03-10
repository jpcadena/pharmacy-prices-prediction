"""
Main script
"""
import logging

import pandas as pd
from numpy import float16, uint16

from analysis import numerical_eda, visualize_data
from core import logging_config
from core.config import NUMERICS
from engineering.extraction import extract_raw_data
from engineering.persistence_manager import PersistenceManager
from models.models import iterate_models, iterate_nn_models

logging_config.setup_logging()
logger: logging.Logger = logging.getLogger(__name__)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


def main() -> None:
    """
    Main function to execute
    :return: None
    :rtype: NoneType
    """
    logger.info("Running main method")

    # TODO: Refactor functionality by removing duplicated code

    d_types: dict = {
        'drug_id': str, 'description': str,
        'administrative_status': 'category', 'marketing_status': 'category',
        'dosage_form': str, 'route_of_administration': str,
        'marketing_authorization_status': 'category',
        'marketing_authorization_process': 'category',
        'pharmaceutical_companies': str}
    drugs_test: pd.DataFrame = extract_raw_data(
        filename='drugs_test.csv', d_types=d_types,
        parse_dates=['marketing_declaration_date',
                     'marketing_authorization_date'])
    d_types['price'] = float16
    drugs_train: pd.Dataframe = extract_raw_data(
        filename='drugs_train.csv', d_types=d_types,
        parse_dates=['marketing_declaration_date',
                     'marketing_authorization_date'])
    numerical_eda(drugs_train)

    active_ingredients: pd.DataFrame = extract_raw_data(
        filename='active_ingredients.csv')
    drug_label_feature_eng: pd.DataFrame = extract_raw_data(
        filename='drug_label_feature_eng.csv')
    numeric_cols = drug_label_feature_eng.select_dtypes(
        include=NUMERICS).columns
    drug_label_feature_eng[numeric_cols] = drug_label_feature_eng[
        numeric_cols].astype('uint8')
    dataframe: pd.DataFrame = drugs_train.merge(
        active_ingredients, on="drug_id")
    dataframe = dataframe.merge(drug_label_feature_eng, on="description")
    dataframe = dataframe.dropna(axis=1)
    dataframe = dataframe.drop(["drug_id", "description"], axis=1)

    visualize_data(drugs_train)

    dataframe = pd.get_dummies(
        dataframe,
        columns=[
            "dosage_form", "pharmaceutical_companies",
            "active_ingredient", "route_of_administration",
            "administrative_status", "marketing_status",
            "approved_for_hospital_use", "marketing_authorization_status",
            "marketing_authorization_process"
        ],
        prefix=[
            "dosage", "pharma", "act_ing", "route",
            "admin_status", "market_status", "hospital_use",
            "ma_status", "ma_process"
        ],
        prefix_sep='_',
        dtype=uint16
    )

    # Fixme: Undersample with numerical values
    # dataframe_undersampled: pd.DataFrame = undersample_data(
    #     dataframe, 'administrative_status')

    dataframe["marketing_declaration_year"] = dataframe[
        "marketing_declaration_date"].dt.year.astype("uint16")
    dataframe["marketing_authorization_year"] = dataframe[
        "marketing_authorization_date"].dt.year.astype("uint16")
    dataframe = dataframe.drop(
        ["marketing_declaration_date", "marketing_authorization_date"], axis=1)
    print(dataframe.shape)
    PersistenceManager.save_to_pickle(dataframe)
    saved: bool = PersistenceManager.save_to_csv(dataframe)
    print(saved)
    numerical_eda(dataframe)

    print(drugs_test.shape)
    df_test: pd.DataFrame = drugs_test.merge(active_ingredients, on="drug_id")
    df_test = df_test.merge(drug_label_feature_eng, on="description")
    df_test = df_test.dropna(axis=1)
    df_test = df_test.drop(["drug_id", "description"], axis=1)
    df_test = pd.get_dummies(
        df_test,
        columns=[
            "dosage_form", "pharmaceutical_companies",
            "active_ingredient", "route_of_administration",
            "administrative_status", "marketing_status",
            "approved_for_hospital_use", "marketing_authorization_status",
            "marketing_authorization_process"
        ],
        prefix=[
            "dosage", "pharma", "act_ing", "route",
            "admin_status", "market_status", "hospital_use",
            "ma_status", "ma_process"
        ],
        prefix_sep='_',
        dtype=uint16
    )
    df_test["marketing_declaration_year"] = df_test[
        "marketing_declaration_date"].dt.year.astype("uint16")
    df_test["marketing_authorization_year"] = df_test[
        "marketing_authorization_date"].dt.year.astype("uint16")
    df_test = df_test.drop(
        ["marketing_declaration_date", "marketing_authorization_date"], axis=1)
    print(df_test.shape)
    iterate_models(dataframe)
    iterate_nn_models(dataframe)


if __name__ == '__main__':
    logger.info("First log message")
    main()
    logger.info("End of the program execution")
