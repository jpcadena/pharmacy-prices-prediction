"""
Main script
"""
import logging
import pandas as pd
from analysis import numerical_eda, visualize_data
from core import logging_config
from engineering import transform_data
from engineering.extraction import extract_raw_data
from engineering.persistence_manager import PersistenceManager

logging_config.setup_logging()
logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main function to execute
    :return: None
    :rtype: NoneType
    """
    logger.info("Running main method")
    dataframe: pd.Dataframe = extract_raw_data()
    try:
        numerical_eda(dataframe)
    except Exception as exc:
        logger.error(exc)
    visualize_data(dataframe)
    PersistenceManager.save_to_pickle(dataframe)
    dataframe = transform_data(dataframe)
    print(dataframe)


if __name__ == '__main__':
    logger.info("First log message")
    main()
    logger.info("End of the program execution")
