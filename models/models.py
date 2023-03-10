"""
Models iteration script
"""
import logging
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, \
    GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from core.decorators import with_logging, benchmark
from engineering.persistence_manager import save_model
from modelling.evaluation import evaluate_model
from modelling.modelling import predict_model
from models.tf_nn import model_nn

logger: logging.Logger = logging.getLogger(__name__)


@with_logging
@benchmark
def iterate_models(
        dataframe: pd.DataFrame, target_column: str = 'price', gpu: bool = True
) -> None:
    """
    Iterates through a list of machine learning models and evaluates
     their performance on the input data
    :param dataframe: A pandas DataFrame containing the input data
    :type dataframe: pd.DataFrame
    :param target_column: The name of the target column in the input
     data
    :type target_column: str
    :param gpu: True if GPU is available to use, False otherwise
    :type gpu: bool
    :return: None
    :rtype: NoneType
    """
    boost_obj: list
    if gpu:
        boost_obj = [
            XGBRegressor(tree_method='gpu_hist', gpu_id=0),
            CatBoostRegressor(task_type="GPU", devices='0'),
            LGBMRegressor(
                device='gpu', gpu_platform_id=0, gpu_device_id=0)
        ]
    else:
        boost_obj = [XGBRegressor(), CatBoostRegressor(), LGBMRegressor()]
    models: list = [
        LinearRegression(), DecisionTreeRegressor(),
        KNeighborsRegressor(), AdaBoostRegressor(),
        GradientBoostingRegressor(), SGDRegressor(), SVR(),
        RandomForestRegressor(),
    ]
    models.extend(boost_obj)
    model_names: list[str] = []
    boost_models: list[bool] = []
    for model in models:
        if isinstance(
                model, (XGBRegressor, CatBoostRegressor, LGBMRegressor)):
            model_names.append(model.__class__.__name__)
            boost_models.append(True)
        else:
            model_names.append(type(model).__name__)
            boost_models.append(False)
    for model, model_name, boost in zip(models, model_names, boost_models):
        print('\n\n', model_name)
        logger.info(model_name)
        y_pred, y_test = predict_model(
            dataframe, model, target_column, boost)
        evaluate_model(y_pred, y_test)
        save_model(model, model_name)


def iterate_nn_models(
        dataframe: pd.DataFrame, target_column: str = 'price', gpu: bool = True
) -> None:
    """
    Iterate Neuronal Network Models
    :param dataframe: The dataframe to use
    :type dataframe: pd.DataFrame
    :param target_column: The target column to predict. The default is
     'price'
    :type target_column: str
    :param gpu: True if GPU is available to use, False otherwise. The
    default is True
    :type gpu: bool
    :return: None
    :rtype: NoneType
    """
    for layer in ['LSTM', 'GRU']:
        print(layer)
        logger.info(layer)
        model_nn(dataframe, target_column, layer, gpu)
