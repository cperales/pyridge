from .nc_elm import NegativeCorrelationELM
from .nc_nn import NegativeCorrelationNN
from .nc_rnn import NegativeCorrelationRNN

nc_algorithm = {
    'NegativeCorrelationELM': NegativeCorrelationELM,
    'NegativeCorrelationNN': NegativeCorrelationNN,
    'NegativeCorrelationRNN': NegativeCorrelationRNN,
}
