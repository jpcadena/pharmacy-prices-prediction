"""
Analysis package initialization
"""
import logging
import pandas as pd
from analysis.analysis import analyze_dataframe
from analysis.visualization import plot_count, plot_distribution, \
    boxplot_dist, plot_scatter, plot_heatmap

logger: logging.Logger = logging.getLogger(__name__)


def numerical_eda(dataframe: pd.DataFrame) -> None:
    """
    EDA based on numerical values for dataset
    :param dataframe: Dataframe to analyze
    :type dataframe: pd.DataFrame
    :return: None
    :rtype: NoneType
    """
    logger.info("Running Exploratory Data Analysis")
    analyze_dataframe(dataframe)


def visualize_data(dataframe: pd.DataFrame) -> None:
    """
    Basic visualization of the dataframe
    :param dataframe: Dataframe to visualize
    :type dataframe: pd.DataFrame
    :return: None
    :rtype: NoneType
    """
    logger.info("Running visualization")
    plot_heatmap(dataframe)
    plot_count(dataframe, ['price', 'reimbursement_rate'])
    plot_distribution(dataframe['price'], 'lightskyblue')
    boxplot_dist(dataframe, 'reimbursement_rate', 'price')
    plot_scatter(dataframe, 'reimbursement_rate', 'price')
