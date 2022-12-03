import logging

from .graphs import Graph
from .ntu_feeder import NTU_Feeder, NTU_Location_Feeder
from .sbu_feeder import SBU_Feeder

__data_args = {
    'ntu': {'class': 60, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
    'ntu120': {'class': 120, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
    'ntu_mutual': {'class': 11, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
    'ntu120_mutual': {'class': 26, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
    'sbu': {'class': 8, 'feeder': SBU_Feeder},
}

def create(dataset, **kwargs):
    g = Graph(dataset, **kwargs)
    try:
        data_args = __data_args[dataset]
        num_class = data_args['class']
    except:
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()

    feeders = {
        'train': data_args['feeder'](dataset=dataset, phase='train', connect_joint=g.connect_joint, **kwargs),
        'eval' : data_args['feeder'](dataset=dataset, phase='eval', connect_joint=g.connect_joint, **kwargs),
    }
    data_shape = feeders['train'].datashape
    if 'ntu' in dataset:
        feeders.update({'location': NTU_Location_Feeder(data_shape)})
    return feeders, data_shape, num_class, g.A, g.parts
