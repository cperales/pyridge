from .linear import RidgeRegressor
from .boosting import BoostingRidgeRegressor
from .ols import OLS

linear_dict = {
    'OrdinaryLeastSquares': OLS,
    'OLS': OLS,
    'LinearRegressor': RidgeRegressor,
    'RidgeRegressor': RidgeRegressor,
    'BoostingRidge': BoostingRidgeRegressor,
}
