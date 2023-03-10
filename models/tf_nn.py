"""
Neural Network using TensorFlow script
"""
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, LSTM, GRU
from keras.models import Sequential

from engineering.persistence_manager import save_model
from modelling.evaluation import evaluate_model
from modelling.train import training
from core.decorators import with_logging, benchmark

logger: logging.Logger = logging.getLogger(__name__)


@with_logging
@benchmark
def predict_nn(
        dataframe: pd.DataFrame, target_column: str, layer: str,
        gpu: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict the output for the neural network using a pandas DataFrame
    :param dataframe: The pandas DataFrame with the target column
    :type dataframe: pd.DataFrame
    :param target_column: The target column name
    :type target_column: str
    :param layer: The type of layer to use (LSTM or GRU)
    :type layer: str
    :param gpu: True if GPU is available to use, False otherwise
    :type gpu: bool
    :return: A tuple with the predicted output and the true output
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    if gpu:
        print(tf.test.is_built_with_cuda())
        print(tf.config.list_physical_devices('GPU'))
        print("Num GPUs Available: ",
              len(tf.config.list_physical_devices('GPU')))
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(8)
        session = tf.compat.v1.Session()
        K.set_session(session)
        tf.config.experimental.set_visible_devices(
            [tf.config.experimental.list_physical_devices('GPU')[0]], 'GPU')
    sequential: Sequential = Sequential()
    x_train, x_test, y_train, y_test = training(dataframe, target_column)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    input_shape = input_shape = (x_train.shape[1], 1)
    if layer == 'LSTM':
        sequential.add(
            LSTM(100, input_shape=input_shape,
                 return_sequences=False))
    elif layer == 'GRU':
        sequential.add(
            GRU(100, input_shape=input_shape,
                return_sequences=False))
    sequential.add(Dense(1, activation='linear'))
    sequential.compile(loss='mean_squared_error', optimizer='adam',
                       metrics=['mean_absolute_error', 'mean_squared_error'])
    sequential.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)
    y_pred: np.ndarray = sequential.predict(x_test)
    logger.info("NN y_pred: %s", y_pred)
    logger.info("NN y_test: %s", y_test)
    save_model(sequential, layer)
    return y_pred, y_test


@with_logging
def model_nn(
        dataframe: pd.DataFrame, target_column: str, layer: str, gpu: bool
) -> None:
    """
    Model Neural Network
    :param dataframe: The pandas DataFrame with the target column
    :type dataframe: pd.DataFrame
    :param target_column: The target column name
    :type target_column: str
    :param layer: The type of layer to use (LSTM or GRU)
    :type layer: str
    :param gpu: True if GPU is available to use, False otherwise
    :type gpu: bool
    :return: None
    :rtype: NoneType
    """
    y_pred_lstm, y_test = predict_nn(dataframe, target_column, layer, gpu)
    evaluate_model(y_pred_lstm, y_test)
