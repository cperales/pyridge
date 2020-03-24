from pyridge.preprocess.prepare import prepare_data
import numpy as np


def test_prepare_classification():
    folder_dataset = 'data/iris'
    train_dataset = 'train_iris.0'
    test_dataset = 'test_iris.0'
    classification = True

    train_data, train_target, train_data_scaler, train_target_scaler = \
        prepare_data(folder=folder_dataset,
                     dataset=train_dataset,
                     sep='\s+',
                     classification=classification)
    test_data, test_target, test_data_scaler, test_target_scaler = \
        prepare_data(folder=folder_dataset,
                     dataset=test_dataset,
                     sep='\s+',
                     data_scaler=train_data_scaler,
                     classification=classification,
                     target_scaler=train_target_scaler)

    if train_target_scaler is not None:
        raise ValueError('In classification problems, '
                         'target is not scaled')

    if train_target_scaler != test_target_scaler or \
            train_data_scaler != test_data_scaler:
        raise ValueError('Scaler must not be changed from '
                         'train to test datasets')

    train_mean = np.mean(train_data, axis=0)
    np.testing.assert_allclose(train_mean,
                               np.zeros_like(train_mean),
                               atol=10**(-14))

    train_std = np.std(train_data, axis=0)
    np.testing.assert_allclose(train_std,
                               np.ones_like(train_std),
                               atol=10 ** (-14))


def test_prepare_regression():
    folder_dataset = 'data_regression/housing'
    train_dataset = 'train_housing.0'
    test_dataset = 'test_housing.0'
    classification = False

    train_data, train_target, train_data_scaler, train_target_scaler = \
        prepare_data(folder=folder_dataset,
                     dataset=train_dataset,
                     sep='\s+',
                     classification=classification)
    test_data, test_target, test_data_scaler, test_target_scaler = \
        prepare_data(folder=folder_dataset,
                     dataset=test_dataset,
                     sep='\s+',
                     data_scaler=train_data_scaler,
                     classification=classification,
                     target_scaler=train_target_scaler)

    if train_target_scaler is None:
        raise ValueError('In regression problems, '
                         'target is scaled')

    if train_target_scaler != test_target_scaler or \
            train_data_scaler != test_data_scaler:
        raise ValueError('Scaler must not be changed from '
                         'train to test datasets')

    train_data_mean = np.mean(train_data, axis=0)
    np.testing.assert_allclose(train_data_mean,
                               np.zeros_like(train_data_mean),
                               atol=10 ** (-14))

    train_std = np.std(train_data, axis=0)
    np.testing.assert_allclose(train_std,
                               np.ones_like(train_std),
                               atol=10 ** (-14))

    train_target_min = train_target.min()
    train_target_max = train_target.max()
    np.testing.assert_allclose([train_target_min, train_target_max],
                               [0.0, 1.0],
                               atol=10 ** (-14))
