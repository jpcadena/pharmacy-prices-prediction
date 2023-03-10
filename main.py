"""
Main script
"""
import logging
import pandas as pd
from analysis import numerical_eda, visualize_data
from core import logging_config
from engineering.extraction import extract_main_data, extract_raw_data
from modelling.preprocessing import numeric_features, preprocess, \
    feature_engineering
from models.models import iterate_models, iterate_nn_models

logging_config.setup_logging()
logger: logging.Logger = logging.getLogger(__name__)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


def main() -> None:
    """
    Main method to execute
    :return: None
    :rtype: NoneType
    """
    logger.info("Running main method")
    drugs_train: pd.Dataframe = extract_main_data(train=True)
    numerical_eda(drugs_train)
    visualize_data(drugs_train)
    active_ingredients: pd.DataFrame = extract_raw_data()
    drug_label_feature_eng: pd.DataFrame = extract_raw_data(
        filename='drug_label_feature_eng.csv')
    drug_label_feature = numeric_features(drug_label_feature_eng)
    preprocessed_df: pd.DataFrame = preprocess(
        drugs_train, active_ingredients, drug_label_feature)
    visualize_data(drugs_train)
    train_df = feature_engineering(preprocessed_df)
    logger.info(train_df.shape)
    drugs_test: pd.DataFrame = extract_main_data(filename='drugs_test.csv')
    preprocessed_test_df: pd.DataFrame = preprocess(
        drugs_test, active_ingredients, drug_label_feature)
    logger.info(preprocessed_test_df.shape)
    iterate_models(train_df)
    iterate_nn_models(train_df)


if __name__ == '__main__':
    logger.info("First log message")
    main()
    logger.info("End of the program execution")
