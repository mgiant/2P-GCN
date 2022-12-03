from . import STGCN_18
from . import TPGCN

__models = {
    'STGCN': STGCN_18,
    '2PGCN': TPGCN,
}

def create(model_type, **kwargs):
    model_name = model_type.split('-')[0]
    return __models[model_name].create(model_type, **kwargs)
