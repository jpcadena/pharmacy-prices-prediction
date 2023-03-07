"""
Models iteration script
"""
import logging
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from modelling.evaluation import evaluate_model
from modelling.modelling import predict_model
from core.decorators import with_logging, benchmark

logger: logging.Logger = logging.getLogger(__name__)


@with_logging
@benchmark
def iterate_models(bow: np.ndarray, dataframe: pd.DataFrame,
                   target_column: str = 'insecurity'):
    """
    Iterates through a list of machine learning models and evaluates
     their performance on the input data
    :param bow:The Array with the features'
    :type bow: np.ndarray
    :param dataframe: A pandas DataFrame containing the input data
    :type dataframe: pd.DataFrame
    :param target_column: The name of the target column in the input
     data
    :type target_column: str
    :return: None
    :rtype: NoneType
    """
    models: list = [LogisticRegression(), SVC(), RandomForestRegressor(),
                    MultinomialNB(), DecisionTreeRegressor(),
                    KNeighborsRegressor(),
                    AdaBoostRegressor(),
                    XGBRegressor(tree_method='gpu_hist', gpu_id=0),
                    CatBoostRegressor(task_type="GPU", devices='0'),
                    LGBMRegressor(
                        device='gpu', gpu_platform_id=0, gpu_device_id=0)]
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
            bow, dataframe, model, target_column, boost)
        evaluate_model(y_pred, y_test)
