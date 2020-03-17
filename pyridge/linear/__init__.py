from .linear import LinearRegressor
from .boosting import BoostingRidgeRegressor
from .ols import OLS

linear_dict = {
    'OrdinaryLeastSquares': OLS,
    'OLS': OLS,
    'LinearRegressor': LinearRegressor,
    'RidgeRegressor': LinearRegressor,
    'BoostingRidge': BoostingRidgeRegressor,
}
