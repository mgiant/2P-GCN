import logging

from .ntu_reader import NTU_Reader


__generator = {
    'ntu_mutual': NTU_Reader,
    'ntu120_mutual': NTU_Reader,
    'ntu': NTU_Reader,
    'ntu120': NTU_Reader,
}

def create(args):
    dataset = args.dataset.split('-')[0]
    dataset_args = args.dataset_args
    if dataset not in __generator.keys():
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()
    return __generator[dataset](args, **dataset_args)
