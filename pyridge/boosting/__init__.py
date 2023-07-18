from .generalized_global import GeneralizedGlobalBRELM
from .boosting_ridge import BoostingRidgeELM
from .kernel_br import KernelBoostingRidgeELM
from .generalized_boosting_ridge import GeneralizedBRELM

boosting_dict = {
    'BoostingRidgeELM': BoostingRidgeELM,
    'KernelBoostingRidgeELM': KernelBoostingRidgeELM,
    'GeneralizedGlobalBRELM': GeneralizedGlobalBRELM,
    'GeneralizedBRELM': GeneralizedBRELM,
}
