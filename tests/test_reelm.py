from pyridge.experiment.test import test_algorithm


def test_reelm():
    hyperparameter_reg = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-1, 2)],
                          'r': [0.2 * i for i in range(1, 3)],
                          'hidden_neurons': [10],
                          'size': [5]}
    algorithm = 'RegularizedEnsembleELM'

    test_algorithm(folder='data',
                   dataset='iris',
                   algorithm=algorithm,
                   hyperparameter=hyperparameter_reg,
                   metric_list=['accuracy', 'rmse'])


def test_reelm_regression():
    hyperparameter_reg = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-1, 2)],
                          'r': [0.2 * i for i in range(1, 3)],
                          'hidden_neurons': [10],
                          'size': [5]}
    algorithm = 'RegularizedEnsembleELM'

    test_algorithm(folder='data_regression',
                   dataset='housing',
                   algorithm=algorithm,
                   hyperparameter=hyperparameter_reg,
                   metric_list=['rmse'],
                   classification=False)


if __name__ == '__main__':
    test_reelm()
    test_reelm_regression()
