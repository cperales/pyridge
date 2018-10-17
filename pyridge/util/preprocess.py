import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler_dict = {'standard': StandardScaler(),
               'min_max': MinMaxScaler()}


def prepare_data(folder,
                 dataset,
                 sep='\s+',
                 scaler='standard',
                 target_end_position=True):
    """

    :param str folder: name of the folder where
        the dataset is.
    :param str dataset: name of the dataset to load.
    :param str sep: separator in string form. Default is spaces.
    :param scaler:
    :param target_end_position: True if target is in the
        last column, False if target is in the first column,
        None if there is no target (pure prediction). Default
        is True.
    :return:
    """
    file_name = os.path.join(folder, dataset)
    file = pd.read_csv(file_name,
                       sep=sep,
                       header=None)

    file_matrix = file.as_matrix()

    if target_end_position is True:
        file_matrix_t = file_matrix.transpose()
        target = file_matrix_t[-1].transpose()
        data = file_matrix_t[:-1].transpose()

    elif target_end_position is False:
        file_matrix_t = file_matrix.transpose()
        target = file_matrix_t[0].transpose()
        data = file_matrix_t[1:].transpose()

    elif target_end_position is None:
        data = file_matrix
        target = None

    else:
        raise ValueError('target_end_position needs'
                         ' to be specified')

    if isinstance(scaler, str):
        scaler = scaler_dict[scaler.lower()].fit(data)
    else:  # Already an object
        pass

    data = scaler.transform(data)

    return data, target, scaler
