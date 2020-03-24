from pyridge.preprocess import scaler_dict
import numpy as np


def test_standard_scaler():
    scaler = scaler_dict['standard']()
    X = np.random.random((50, 3)) * 2.0 - 1.0
    X_scaled = scaler.fit_transform(X)

    X_scaled_mean = np.mean(X_scaled, axis=0)
    np.testing.assert_allclose(X_scaled_mean,
                               np.zeros_like(3),
                               atol=10**(-14))

    X_scaled_std = np.std(X_scaled, axis=0)
    np.testing.assert_allclose(X_scaled_std,
                               np.ones_like(3),
                               atol=10 ** (-14))

    X_1 = scaler.inverse_transform(X_scaled)
    np.testing.assert_allclose(X,
                               X_1,
                               atol=10 ** (-14))


def test_min_max_scaler():
    scaler = scaler_dict['min_max']()
    X = np.random.random((50, 3)) * 2.0 - 1.0
    X_scaled = scaler.fit_transform(X)

    X_scaled_mean = np.min(X_scaled, axis=0)
    np.testing.assert_allclose(X_scaled_mean,
                               np.zeros_like(3),
                               atol=10**(-14))

    X_scaled_max = np.max(X_scaled, axis=0)
    np.testing.assert_allclose(X_scaled_max,
                               np.ones_like(3),
                               atol=10 ** (-14))

    X_1 = scaler.inverse_transform(X_scaled)
    np.testing.assert_allclose(X,
                               X_1,
                               atol=10 ** (-14))


def test_log_scaler():
    scaler = scaler_dict['log']()
    X = np.random.random((50, 3)) * 2.0 - 1.0
    X_scaled = scaler.fit_transform(X)

    X_scaled_min = X_scaled.min()
    if X_scaled_min < 0.0:
        raise ValueError('Logatimic values are'
                         'greater than 0')

    X_1 = scaler.inverse_transform(X_scaled)
    np.testing.assert_allclose(X,
                               X_1,
                               atol=10 ** (-14))
