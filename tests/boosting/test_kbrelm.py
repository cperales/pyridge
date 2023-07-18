from pyridge.experiment.check import check_algorithm, create_logger


def test_kernel_boosting_ridge():
    hyperparameter = {'kernel': ['rbf', 'linear'],
                      'reg': [10 ** i for i in range(-1, 2)],
                      'gamma': [10],
                      'size': [5]}
    value_dict = check_algorithm(folder='data',
                                 dataset='iris',
                                 algorithm='KernelBoostingRidgeELM',
                                 hyperparameter=hyperparameter,
                                 metric_list=['accuracy', 'rmse'])


def test_kernel_boosting_ridge_regression():
    hyperparameter = {'kernel': ['rbf', 'linear'],
                      'reg': [10 ** i for i in range(-1, 2)],
                      'gamma': [10],
                      'size': [5]}
    value_dict = check_algorithm(folder='data_regression',
                                 dataset='housing',
                                 algorithm='KernelBoostingRidgeELM',
                                 hyperparameter=hyperparameter,
                                 metric_list=['rmse'],
                                 classification=False)


if __name__ == '__main__':
    create_logger('INFO')
    test_kernel_boosting_ridge()
    test_kernel_boosting_ridge_regression()