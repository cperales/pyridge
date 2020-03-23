class Scaler:
    """
    Scaler for data and target, similar to others
    from sklearn but avoiding shape restrictions.

    The intention of this object is to be inherit.
    """
    def get_params(self):
        return self.__dict__

    def fit(self, values):
        pass

    def transform(self, values):
        return values

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        return values
