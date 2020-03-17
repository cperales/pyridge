from pyridge.experiment.check import check_algorithm


def test_linear():
    hyperparameter_elm = {'reg': [10 ** i for i in range(-3, 4)]}
    value_dict = check_algorithm(folder='data',
                                 dataset='iris',
                                 algorithm='LinearRegressor',
                                 hyperparameter=hyperparameter_elm,
                                 metric_list=['accuracy', 'rmse'])


def test_ols():
    value_dict = check_algorithm(folder='data',
                                 dataset='iris',
                                 algorithm='OLS',
                                 hyperparameter=None,
                                 metric_list=['accuracy', 'rmse'])


def test_linear_regression():
    hyperparameter_elm = {'reg': [10 ** i for i in range(-2, 3)]}
    value_dict = check_algorithm(folder='data_regression',
                                 dataset='housing',
                                 algorithm='LinearRegressor',
                                 hyperparameter=hyperparameter_elm,
                                 metric_list=['rmse'],
                                 classification=False)


def test_ols_regression():
    value_dict = check_algorithm(folder='data_regression',
                                 dataset='housing',
                                 algorithm='OLS',
                                 hyperparameter=None,
                                 metric_list=['rmse'],
                                 classification=False)
