import os
import pandas as pd
from pyelm.utils.target_encode import j_encode
from sklearn import preprocessing


def prepare_data(folder,
                 dataset,
                 n_targ=None,
                 header=None,
                 sep='\s+',
                 j_encoding=True):
    """

    :param str folder: name of the folder where
        the dataset is.
    :param str dataset: name of the dataset to load.
    :param int n_targ: number of labels to classify.
    :param header:
    :param sep:
    :param bool j_encoding:
    :return:
    """
    train_file_name = os.path.join(folder, dataset)
    train_file = pd.read_csv(train_file_name,
                             sep=sep,
                             header=header)
    train_file_matrix = train_file.as_matrix()
    train_file_matrix_t = train_file_matrix.transpose()
    train_target = train_file_matrix_t[-1].transpose()
    train_data = train_file_matrix_t[:-1].transpose()
    train_data = preprocessing.scale(train_data)

    if j_encoding is True:
        train_target = j_encode(train_target, n_targ=n_targ)

    return train_data, train_target
