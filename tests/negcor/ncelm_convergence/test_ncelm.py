from pyridge.negcor.nc_elm import NegativeCorrelationELM
from pyridge.util import prepare_data, metric_dict


def test_ncelm():
    # Select one k-fold of the data (0), load train and test folds and scale
    train_data, train_target, data_scaler, target_scaler = \
        prepare_data(folder='data/qsar-biodegradation',
                     dataset='train_qsar-biodegradation.0')
    test_data, test_target, _, _ = prepare_data(folder='data/qsar-biodegradation',
                                                dataset='test_qsar-biodegradation.0',
                                                data_scaler=data_scaler,
                                                target_scaler=target_scaler)

    for lambda_ in [0.00001, 0.0001, 0.001]:  # For a range of lambda values
        parameter = {'activation': 'sigmoid',
                     'hidden_neurons': 50,
                     'reg': 1.0,
                     'lambda_': lambda_,
                     'max_iter_': 5,
                     'size': 25}

        clf = NegativeCorrelationELM()
        clf.fit(train_data=train_data,
                train_target=train_target,
                parameter=parameter)

        acc = metric_dict['accuracy'](clf, test_data, test_target)
        print('C =', 1.0, ', lambda =', lambda_, ', accuracy =', acc)


if __name__ == '__main__':
    test_ncelm()
