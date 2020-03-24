import os
import pandas as pd
from .min_max import MinMaxScaler
from .standard import StandardScaler


scaler_dict = {
    'min_max': MinMaxScaler,
    'standard': StandardScaler
}


def prepare_data(folder,
                 dataset,
                 sep='\s+',
                 data_scaler='standard',
                 classification=True,
                 target_scaler='min_max'):
    """
    Read the data from the files and scale them.
    Target is assumed to be the last column.

    :param str folder: name of the folder where
        the dataset is.
    :param str dataset: name of the dataset to load.
    :param str sep: separator in string form. Default is spaces.
    :param data_scaler: data must be to be scaled.
    :param bool classification: if `classification` is True, then
        target needs to be scaled.
    :param target_scaler: if classification is true, target
        is scaled according to this parameter.
    :return: data, target, scaler_data, scaler_target
    """
    file_name = os.path.join(folder, dataset)
    file = pd.read_csv(file_name,
                       sep=sep,
                       header=None)

    data = file[file.columns[:-1]].values
    target = file[file.columns[-1]].values

    # Data
    data_scaler, data = \
        scale_values(scaler=data_scaler, values=data)

    # Target
    if classification is False:
        target_scaler, target = \
            scale_values(scaler=target_scaler, values=target)
    else:
        target_scaler = None

    return data, target, data_scaler, target_scaler


def scale_values(scaler, values):
    """
    Function to instance an scaler (if necessary) and
    scale the values.
    """
    if isinstance(scaler, str):
        scaler = scaler_dict[scaler.lower()]()
        scaled_values = scaler.fit_transform(values)
    else:
        scaled_values = scaler.transform(values)
    return scaler, scaled_values

