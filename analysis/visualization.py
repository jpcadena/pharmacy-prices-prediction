"""
Visualization script
"""
import logging
import re
from typing import Optional

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from core.config import FIG_SIZE, FONT_SIZE, PALETTE, RE_PATTERN, RE_REPL
from engineering.persistence_manager import DataType

logger: logging.Logger = logging.getLogger(__name__)


def plot_count(
        dataframe: pd.DataFrame, variables: list, hue: Optional[str] = None,
        data_type: DataType = DataType.FIGURES
) -> None:
    """
    This method plots the counts of observations from the given variables
    :param dataframe: dataframe containing the raw data
    :type dataframe: pd.DataFrame
    :param variables: list of discrete columns to plot
    :type variables: list
    :param hue:
    :type hue: str
    :param data_type: Path where data will be saved: RAW or
     PROCESSED. The default is FIGURES
    :type data_type: DataType
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    plt.suptitle('Count-plot for Discrete variables')
    plot_iterator: int = 1
    for i in variables:
        plt.subplot(1, 3, plot_iterator)
        if hue:
            sns.countplot(x=dataframe[i], hue=dataframe[hue], palette=PALETTE)
        else:
            sns.countplot(x=dataframe[i], palette=PALETTE)
        label = re.sub(pattern=RE_PATTERN, repl=RE_REPL, string=i)
        plt.xlabel(label, fontsize=15)
        plt.ylabel('Count', fontsize=15)
        plot_iterator += 1
        plt.savefig(f'{data_type.value}discrete_{i}.png')
        plt.show()


def plot_distribution(
        df_column: pd.Series, color: str,
        data_type: DataType = DataType.FIGURES
) -> None:
    """
    This method plots the distribution of the given quantitative continuous
    variable
    :param df_column: Single column
    :type df_column: pd.Series
    :param color: color for the distribution
    :type color: str
    :param data_type: Path where data will be saved: RAW or
     PROCESSED. The default is FIGURES
    :type data_type: DataType
    :return: None
    :rtype: NoneType
    """
    label: str = re.sub(
        pattern=RE_PATTERN, repl=RE_REPL, string=str(df_column.name))
    sns.displot(x=df_column, kde=True, color=color, height=8, aspect=1.875)
    plt.title('Distribution Plot for ' + label)
    plt.xlabel(label, fontsize=FONT_SIZE)
    plt.ylabel('Frequency', fontsize=FONT_SIZE)
    plt.savefig(f'{data_type.value}{str(df_column.name)}.png')
    plt.show()


def boxplot_dist(
        dataframe: pd.DataFrame, first_variable: str, second_variable: str,
        data_type: DataType = DataType.FIGURES) -> None:
    """
    This method plots the distribution of the first variable data
    in regard to the second variable data in a boxplot
    :param dataframe: data to use for plot
    :type dataframe: pd.DataFrame
    :param first_variable: first variable to plot
    :type first_variable: str
    :param second_variable: second variable to plot
    :type second_variable: str
    :param data_type: Path where data will be saved: RAW or
     PROCESSED. The default is FIGURES
    :type data_type: DataType
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    x_label: str = re.sub(
        pattern=RE_PATTERN, repl=RE_REPL, string=first_variable)
    y_label: str = re.sub(
        pattern=RE_PATTERN, repl=RE_REPL, string=second_variable)
    sns.boxplot(x=first_variable, y=second_variable, data=dataframe,
                palette=PALETTE)
    plt.title(x_label + ' in regards to ' + y_label, fontsize=FONT_SIZE)
    plt.xlabel(x_label, fontsize=FONT_SIZE)
    plt.ylabel(y_label, fontsize=FONT_SIZE)
    plt.savefig(
        f'{data_type.value}discrete_{first_variable}_{second_variable}.png')
    plt.show()


def plot_scatter(
        dataframe: pd.DataFrame, x_array: str, y_array: str,
        hue: Optional[str] = None, data_type: DataType = DataType.FIGURES
) -> None:
    """
    This method plots the relationship between x and y for hue subset
    :param dataframe: dataframe containing the data
    :type dataframe: pd.DataFrame
    :param x_array: x-axis column name from dataframe
    :type x_array: str
    :param y_array: y-axis column name from dataframe
    :type y_array: str
    :param hue: grouping variable to filter plot
    :type hue: str
    :param data_type: Path where data will be saved: RAW or
     PROCESSED. The default is FIGURES
    :type data_type: DataType
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    sns.scatterplot(x=x_array, data=dataframe, y=y_array, hue=hue,
                    palette=PALETTE)
    label: str = re.sub(pattern=RE_PATTERN, repl=RE_REPL, string=y_array)
    plt.title(f'{x_array} Wise {label} Distribution')
    print(dataframe[[x_array, y_array]].corr())
    plt.savefig(f'{data_type.value}{x_array}_{y_array}_{hue}.png')
    plt.show()


def plot_heatmap(
        dataframe: pd.DataFrame, data_type: DataType = DataType.FIGURES
) -> None:
    """
    Plot heatmap to analyze correlation between features
    :param dataframe: dataframe containing the data
    :type dataframe: pd.DataFrame
    :param data_type: Path where data will be saved: RAW or
     PROCESSED. The default is FIGURES
    :type data_type: DataType
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    data: pd.DataFrame = dataframe.corr()
    sns.heatmap(data=data, annot=True, cmap="RdYlGn")
    plt.title('Heatmap showing correlations among columns',
              fontsize=FONT_SIZE)
    plt.savefig(f'{data_type.value}correlations_heatmap.png')
    plt.show()
