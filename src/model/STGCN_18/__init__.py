
from re import M
from .st_gcn_aaai18 import ST_GCN_18

def create(model_type, **kwargs):

    kwargs.update({
        'in_channels': kwargs['data_shape'][1],
    })
    return ST_GCN_18(**kwargs)