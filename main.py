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
    print(dataframe.shape)
    print(dataframe.dtypes)

    visualize_data(drugs_train)

    dataframe = pd.get_dummies(
        dataframe, columns=[
            "dosage_form",  # 19
            "pharmaceutical_companies",  # 33
            "active_ingredient",
            "route_of_administration", "administrative_status",
            "marketing_status", "approved_for_hospital_use",
            "marketing_authorization_status",
            "marketing_authorization_process"], dtype=uint16)

    # Fixme: Undersample with numerical values
    # dataframe_undersampled: pd.DataFrame = undersample_data(
    #     dataframe, 'administrative_status')

    dataframe["marketing_declaration_year"] = dataframe[
        "marketing_declaration_date"].dt.year.astype("uint16")
    dataframe["marketing_authorization_year"] = dataframe[
        "marketing_authorization_date"].dt.year.astype("uint16")
    dataframe = dataframe.drop(
        ["marketing_declaration_date", "marketing_authorization_date"], axis=1)
    PersistenceManager.save_to_pickle(dataframe)
    # dataframe = transform_data(dataframe)
    saved: bool = PersistenceManager.save_to_csv(dataframe)
    print(saved)
    numerical_eda(dataframe)
    iterate_models(dataframe)
    iterate_nn_models(dataframe)


if __name__ == '__main__':
    logger.info("First log message")
    main()
    logger.info("End of the program execution")
