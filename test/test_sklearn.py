from pyelm.algorithm.sklearn_svm import SklearnSVC
from pyelm.utils.preprocess import prepare_data
from pyelm.utils import accuracy
import numpy as np
from pyelm.utils import cross_validation


def sklearn_comparison():
    # Data
    folder = '../data/hepatitis'
    train_dataset = 'train_hepatitis.0'
    train_data, train_target = prepare_data(folder=folder,
                                            dataset=train_dataset,
                                            j_encoding=False)
    test_dataset = 'test_hepatitis.0'
    test_data, test_target = prepare_data(folder=folder,
                                          dataset=test_dataset,
                                          j_encoding=False)

    # SVC sklearn
    clf = SklearnSVC()
    clf.fit(X=train_data, y=train_target)
    pred_targ = clf.predict(X=test_data)
    accuracy(pred_targ=pred_targ, real_targ=test_target, j_encoded=False)  # To test it works
    clf.grid_param = dict()
    # clf.set_cv_range = KernelMethod.set_cv_range

    # Hyperparameters
    C_range = [10 ** i for i in range(-2, 3)]
    k_range = [10 ** i for i in range(-2, 3)]
    kernel_fun = 'rbf'
    hyperparameters = {'kernelFun': kernel_fun,
                       'C': C_range,
                       'k': k_range}

    clf.set_cv_range(hyperparameters)
    cross_validation(classifier=clf, X=train_data, y=train_target)
    pred_targ = clf.predict(X=test_data)
    accuracy(pred_targ=pred_targ, real_targ=test_target, j_encoded=False)  # To test it works


if __name__ == '__main__':
    sklearn_comparison()
