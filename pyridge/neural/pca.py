from ..util.activation import activation_dict
from .elm import ELM
import numpy as np
from sklearn.decomposition import PCA


class PCAELM(ELM):
    """
    Principal Component Analysis applied to ELM framework.
    """
    __name__ = 'PCA ELM'
    pca = None
    pca_components = None
    pca_n_components: int
    pca_explained_var_ratio_ = None
    pca_perc: float = 0.9

    def select_components(self):
        """
        Sum the explained variance ration util achieve or
        overpass pca_perc.

        :return:
        """
        var = 0.0
        n_component = 0
        while var < self.pca_perc:
            var += self.pca_explained_var_ratio_[n_component]
            n_component += 1
        self.pca_n_components = n_component

    def get_pca_(self):
        """
        Return the input weight associated with
        Principal Component Analysis.

        :return: pca_input_weight
        """
        self.pca = PCA(n_components=self.dim,
                       copy=True,
                       whiten=False,
                       svd_solver='auto',
                       tol=0.0,
                       iterated_power='auto',
                       random_state=None)
        factor = self.dim / self.n
        if factor > 1.0:
            train_data = np.repeat(self.train_data, int(factor + 1), axis=0)
        else:
            train_data = self.train_data
        self.pca.fit(X=train_data)
        self.pca_components = self.pca.components_
        self.pca_explained_var_ratio_ = self.pca.explained_variance_ratio_
        self.select_components()
        pca_input_weight = self.pca_components[:self.pca_n_components]
        return pca_input_weight

    def get_weight_bias_(self):
        """
        Weight is obtained from PCA, bias is 0.

        At this point of the code, train is already a
        property of the object.
        :return:
        """
        self.neuron_fun = activation_dict[self.activation]
        self.input_weight = self.get_pca_()
        # For consistency with other ELM, these values needs to be reported
        self.bias_vector = 0.0
        self.hidden_neurons = self.input_weight.shape[0]

    def get_h_matrix(self, data):
        """

        :param data:
        :return:
        """
        temp_h_matrix = np.dot(data, self.input_weight.T)
        h_matrix = self.neuron_fun(temp_h_matrix)
        return h_matrix

