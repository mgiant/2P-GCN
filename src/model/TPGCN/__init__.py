import logging

from . import blocks
from .nets import TPGCN
from .modules import ResGCN_Module, AttGCN_Module
from .attentions import *

__attention = {
    'pa': Part_Att,
    'stja': ST_Joint_Att,
    'stpa': ST_Part_Att,
}

__structure = {
    'm19': {'structure': [1,2,3,3], 'spatial_block': 'Basic', 'temporal_block': 'MultiScale'},
    'a19': {'structure': [1,2,3,3], 'spatial_block': 'AAGCN', 'temporal_block': 'MultiScale'},
    'b19': {'structure': [1,2,3,3], 'spatial_block': 'Basic', 'temporal_block': 'Basic'},
    'c19': {'structure': [1,2,3,3], 'spatial_block': 'CTRGCN', 'temporal_block': 'MultiScale'},
}

__reduction = {
    'r1': {'reduction': 1},
    'r2': {'reduction': 2},
    'r4': {'reduction': 4},
    'r8': {'reduction': 8},
}

def create(_, block_structure, att_type, reduction='r1', **kwargs):
    if att_type != 'none':
        kwargs.update({'module': AttGCN_Module, 'attention': __attention[att_type]})
    else:
        kwargs.update({'module': ResGCN_Module, 'attention': None})
    return TPGCN(**(__structure[block_structure]), **(__reduction[reduction]), **kwargs)
