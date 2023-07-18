from pyridge.experiment.check import check_algorithm, create_logger


def test_boosting_ridge():
    hyperparameter_boost = {'activation': ['sigmoid'],
                            'reg': [10 ** i for i in range(-1, 2)],
                            'hidden_neurons': [10],
                            'size': [5]}
    value_dict = check_algorithm(folder='data',
                                 dataset='iris',
                                 algorithm='BoostingRidgeELM',
                                 hyperparameter=hyperparameter_boost,
                                 metric_list=['accuracy', 'rmse'])


def test_boosting_ridge_regression():
    hyperparameter_boost = {'activation': ['sigmoid'],
                            'reg': [10 ** i for i in range(-1, 2)],
                            'hidden_neurons': [10],
                            'size': [5]}
    value_dict = check_algorithm(folder='data_regression',
                                 dataset='housing',
                                 algorithm='BoostingRidgeELM',
                                 hyperparameter=hyperparameter_boost,
                                 metric_list=['rmse'],
                                 classification=False)


if __name__ == '__main__':
    create_logger('INFO')
    test_boosting_ridge()
    test_boosting_ridge_regression()