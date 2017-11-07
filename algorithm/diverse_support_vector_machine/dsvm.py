import pickle
import numpy as np
import matplotlib.pyplot as plt
import logging
from ..classifier import Classifier
from .svm_solver import *
from .generate_data import plot_data_with_labels, generate_gaussian


class DiverseLinearSVM(Classifier):
    """
    Implementation of the diverse linear support vector machine.
    """
    def __init__(self, **kwargs):
        self.c = kwargs['c'] if 'c' in kwargs else 1.0
        self.d = kwargs['d'] if 'd' in kwargs else 1.0
        self.x = None
        self.y = None
        self.w = None
        self.b = None
        self.prev_w = None

    def fit(self, data, target, soft=True, diverse=False):
        """
        Fit the model according to the given training data.
        """
        self.add_x_y(x=data, y=target)
        # Data can be added as a pickle using read_data method
        if diverse is True:
            alphas = fit_diverse(self.x, self.y, self.prev_w)
        else:
            if soft is True:
                alphas = fit_soft(self.x, self.y, self.c)
            else:
                alphas = fit(self.x, self.y)

        # get weights
        w = np.sum(alphas * self.y[:, None] * self.x, axis=0)
        # # get b
        # cond = (alphas > 1e-4).reshape(-1)
        # b = self.y[cond] - np.dot(self.x[cond], w)
        b_vector = self.y - np.dot(self.x, w)
        b = b_vector.sum() / b_vector.size

        # normalize
        norm = np.linalg.norm(w)
        w, b = w / norm, b / norm

        # Self values
        self.w = w
        if self.prev_w is None:
            self.prev_w = w
        # self.b = b
        self.b = b

    def read_data(self, dataname):
        """
        Take values from a pickle.

        :param dataname:
        :return:
        """
        self.x, self.y = read_data(dataname)

    def generate_data(self,
                      m_1=1.0,
                      m_2=2.1,
                      c_1=0.3,
                      c_2=0.2,
                      num=50,
                      dim=2,
                      dataname='gaussiandata.pickle'):
        """
        Generate random binary data and save it into a pickle.

        :param dataname: the name of the pickle with the data.
        :return:
        """
        m_1 = m_1 * np.ones((dim,))
        m_2 = m_2 * np.ones((dim,))
        c_1 = np.diag(c_1 * np.ones((dim,)))
        c_2 = np.diag(c_2 * np.ones((dim,)))

        # generate points for class 1
        x1 = generate_gaussian(m_1, c_1, num)
        # generate points for class 2
        x2 = generate_gaussian(m_2, c_2, num)
        # labels
        y1 = np.ones((x1.shape[0],))
        y2 = -np.ones((x2.shape[0],))
        # join
        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
        logging.debug('x {} y {}'.format(x.shape, y.shape))
        # plot_data_with_labels(x, y)
        # write
        with open(dataname, 'wb') as f:
            pickle.dump((x, y), f)
        # Instance
        self.x = x
        self.y = y

    def add_x_y(self, x, y):
        """
        Add the arrays directly.

        :param x:
        :param y:
        :return:
        """
        self.x, self.y = x, y

    def plot_data_separator(self, figname='dvsm.png'):
        """
        Plot the data and the vector.

        :param figname: Name in order to save the fig.
        """
        if not 'fig' in self.__dict__:
            self.fig, self.ax = plt.subplots()
            # plot_data_with_labels(self.x, self.y, self.ax)
            plot_data_with_labels(self.x, self.y)
        plot_separator(self.ax, self.w, self.b)
        if self.prev_w is not None:
            plot_separator(self.ax, self.prev_w, self.b)
        plt.savefig(figname)
        # plt.show()
        # plt.close()

    def classify(self, data):
        """
        Function to classify an array of elements to classify.

        :param data:
        """
        targets = np.array([np.dot(self.w, elem) - self.b for elem in data])
        return np.sign(targets)
