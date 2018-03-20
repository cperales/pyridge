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
    :param list header: header list.
    :param str sep: separator in string form. Default is spaces.
    :param bool j_encoding: target needs to be encoded or not.
    :return:
    """
    file_name = os.path.join(folder, dataset)
    file = pd.read_csv(file_name,
                       sep=sep,
                       header=header)
    file_matrix = file.as_matrix()
    file_matrix_t = file_matrix.transpose()
    target = file_matrix_t[-1].transpose()
    data = file_matrix_t[:-1].transpose()
    data = preprocessing.scale(data)

    if j_encoding is True:
        target = j_encode(target, n_targ=n_targ)

    return data, target
