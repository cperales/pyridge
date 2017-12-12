from .classifier import Classifier


class CVMethod(Classifier):

    C = 0
    ensemble_size = 1
    grid_param = {}

    def config(self, train_data, train_targ):
        # Cross validation
        pass
