import os
import pandas as pd
from pyridge.utils.target_encode import j_encode
from sklearn import preprocessing


def prepare_data(folder,
                 dataset,
                 n_targ=None,
                 header=None,
                 sep='\s+',
                 target_end_position=True,
                 j_encoding=True):
    """

    :param str folder: name of the folder where
        the dataset is.
    :param str dataset: name of the dataset to load.
    :param int n_targ: number of labels to classify.
    :param list header: header list.
    :param str sep: separator in string form. Default is spaces.
    :param target_end_position: True if target is in the
        last column, False if target is in the first column,
        None if there is no target (pure prediction). Default
        is True.
    :param bool j_encoding: target needs to be encoded or not.
    :return:
    """
    file_name = os.path.join(folder, dataset)
    file = pd.read_csv(file_name,
                       sep=sep,
                       header=header)
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

    data = preprocessing.scale(data)

    if j_encoding is True:
        target = j_encode(target, n_targ=n_targ)

    return data, target
