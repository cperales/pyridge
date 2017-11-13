import logging
# from ..classifier import Classifier
from ..kernel_classifier import KernelClassifier
from .svm_solver import *
from .generate_data import plot_data_with_labels, generate_gaussian


class DiverseLinearSVM(KernelClassifier):
    """
    Implementation of the diverse linear support vector machine.
    """
    def __init__(self, **kwargs):
        self.c = kwargs['c'] if 'c' in kwargs else 1.0
        self.d = kwargs['d'] if 'd' in kwargs else 1.0

        if 'kernel' in kwargs and kwargs['kernel'].lower() != 'linear':
            logging.warning('Only linear kernel is allowed for Diverse SVM')
            kwargs['kernel'] = 'linear'

        self.kernel = self.kernel_dict[kwargs['kernel'].lower()]
        self.train_data = None
        self.train_target = None
        self.w = None
        self.b = None
        self.prev_w = None

    def fit(self, data, target, soft=True, diverse=False):
        """
        Fit the model according to the given training data.
        """
        self.add_train(train_data=data, train_target=target)
        # Data can be added as a pickle using read_data method
        if diverse is True:
            alphas = fit_diverse(x=self.train_data, y=self.train_target, w=self.prev_w)
        else:
            if soft is True:
                alphas = fit_soft(x=self.train_data, y=self.train_target, c=self.c, kernel=self.kernel)
            else:
                alphas = fit(self.train_data, self.train_target)

        # get weights
        # alphas = alphas / np.linalg.norm(alphas)
        w = np.sum(alphas * self.train_target[:, None] * self.train_data, axis=0)
        b_vector = self.train_target - np.dot(self.train_data, w)  # np.dot(w, self.train_data.T) == np.dot(self.train_data, w)
        b = np.mean(b_vector)

        # # normalize
        # norm = np.linalg.norm(alphas)
        # w, b = w / norm, b / norm

        # Self values
        self.w = w
        if self.prev_w is None:
            self.prev_w = w
        # self.b = b
        self.b = b

    def read_data(self, data_name):
        """
        Take values from a pickle.

        :param data_name:
        :return:
        """
        self.train_data, self.train_target = read_data(data_name)

    def generate_data(self,
                      m_1=1.0,
                      m_2=2.1,
                      c_1=0.3,
                      c_2=0.2,
                      num=50,
                      dim=2,
                      data_name='gaussiandata.pickle'):
        """
        Generate random binartrain_target data and save it into a pickle.

        :param m_1:
        :param m_2:
        :param c_1:
        :param c_2:
        :param num:
        :param dim:
        :param data_name:
        :return:
        """
        m_1 = m_1 * np.ones((dim,))
        m_2 = m_2 * np.ones((dim,))
        c_1 = np.diag(c_1 * np.ones((dim,)))
        c_2 = np.diag(c_2 * np.ones((dim,)))

        # generate points for class 1
        train_data1 = generate_gaussian(m_1, c_1, num)
        # generate points for class 2
        train_data2 = generate_gaussian(m_2, c_2, num)
        # labels
        train_target1 = np.ones((train_data1.shape[0],))
        train_target2 = -np.ones((train_data2.shape[0],))
        # join
        train_data = np.concatenate((train_data1, train_data2), axis=0)
        train_target = np.concatenate((train_target1, train_target2), axis=0)
        logging.debug('train_data {} train_target {}'.format(train_data.shape, train_target.shape))
        # plot_data_with_labels(train_data, train_target)
        # write
        with open(data_name, 'wb') as f:
            pickle.dump((train_data, train_target), f)
        # Instance
        self.train_data = train_data
        self.train_target = train_target

    def add_train(self, train_data, train_target):
        """
        Add the arrays directltrain_target.

        :param train_data:
        :param train_target:
        :return:
        """
        self.train_data, self.train_target = train_data, train_target

    def plot_data_separator(self, fig_name='dvsm.png'):
        """
        Plot the data and the vector.

        :param fig_name: Name in order to save the fig.
        """
        if not 'fig' in self.__dict__:
            self.fig, self.ax = plt.subplots()
            # plot_data_with_labels(self.train_data, self.train_target, self.ax)
            plot_data_with_labels(self.train_data, self.train_target)
        plot_separator(self.ax, self.w, self.b)
        if self.prev_w is not None:
            plot_separator(self.ax, self.prev_w, self.b)
        plt.savefig(fig_name)
        # plt.show()
        # plt.close()

    def classify(self, data):
        """
        Function to classify an array of elements to classify.

        :param data:
        """
        targets = np.array([self.kernel(self.w, elem) + self.b for elem in data])
        return np.sign(targets)
