import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler_dict = {'standard': StandardScaler(),
               'min_max': MinMaxScaler()}


def prepare_data(folder,
                 dataset,
                 sep='\s+',
                 scaler='standard'):
    """
    Read the data from the files and scale them.
    Target is supposed to be at the last column.

    :param str folder: name of the folder where
        the dataset is.
    :param str dataset: name of the dataset to load.
    :param str sep: separator in string form. Default is spaces.
    :param scaler:
    :return:
    """
    file_name = os.path.join(folder, dataset)
    file = pd.read_csv(file_name,
                       sep=sep,
                       header=None)

    file_matrix = file.as_matrix()

    file_matrix_t = file_matrix.transpose()
    target = file_matrix_t[-1].transpose()
    data = file_matrix_t[:-1].transpose()

    if isinstance(scaler, str):
        scaler = scaler_dict[scaler.lower()].fit(data)
    else:  # Already an object
        pass

    data = scaler.transform(data)

    return data, target, scaler
