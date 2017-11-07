from ..kernel_classifier import KernelClassifier


class KernelELM(KernelClassifier):
    """
    Kernel version of Extreme Learning Machine classifier.
    """
    def __init__(self, train_data, train_target, kernel_type, **args):
        self.train = train
        self.kernel_function = self.kernel[kernel_type]

    def fit(self, train_data, train_target):
        """

        :param train_data:
        :param train_target:
        :return:
        """
        omega_train = self.kernel_function

    def classify(self):
        pass

