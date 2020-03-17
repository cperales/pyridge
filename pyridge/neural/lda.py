from ..util.activation import activation_dict
from .pca import PCAELM
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class PCALDAELM(PCAELM):
    """
    Principal Component Analysis with Linear
    Discriminant Analysis applied to ELM framework.

    ONLY FOR CLASSIFICATION.
    """
    __name__ = 'PCA LDA ELM'
    lda = None
    lda_components = None

    def __init__(self, classification: bool = True, logging: bool = True):
        if classification is False:
            raise ValueError('PCA LDA ELM is only for classification')
        super().__init__(classification=classification, logging=logging)

    def get_lda_(self):
        """
        Return the input weight associated with
        Linear Discriminant Analysis.

        :return:
        """
        self.lda = LinearDiscriminantAnalysis(solver='svd',
                                              shrinkage=None,
                                              priors=None,
                                              # n_components=self.pca_n_components,
                                              n_components=None,
                                              store_covariance=False,
                                              tol=0.0001)
        self.lda.fit(X=self.train_data,
                     y=self.train_target)
        self.lda_components = self.lda.coef_
        return self.lda_components

    def get_weight_bias_(self):
        """
        Weight is obtained from PCA, bias is 0.

        At this point of the code, train is already a
        property of the object.
        :return:
        """
        self.neuron_fun = activation_dict[self.activation]
        # Principal Component Analysis
        self.get_pca_()
        n_component = self.select_components()
        pca_input_weight = self.pca_components[:n_component]
        # Linear Discrimination Analysis
        lda_input_weight = self.get_lda_()
        self.input_weight = np.concatenate([pca_input_weight,
                                            lda_input_weight])
        # For consistency with other ELM, these values
        # needs to be reported
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
