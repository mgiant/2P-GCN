'''
Description: script to calculate mean and std accuracy for each config
Author: jackieysong
Email: jackieysong@tencent.com
Date: 2021-10-14 16:43:06
Acknowledgements: 
'''

import os
import json
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Calculate mean and std accuracy for each config')
parser.add_argument('--config', '-c', type=str, default='basic', help='Using config')
parser.add_argument('--model_name', '-m', type=str, default='EfficientGCN-B0', help='Using model')
parser.add_argument('--benchmark', '-b', type=str, default='ntu-xsub', help='Using model')
args, _ = parser.parse_known_args()

if args.config in ['2002', '2002_ori', '2006', '2006_ori', '2010', '2010_ori']:
    args.benchmark = 'ntu-xview'
if args.config in ['2003', '2007', '2011']:
    args.benchmark = 'ntu-xsub120'
if args.config in ['2004', '2008', '2012']:
    args.benchmark = 'ntu-xset120'

if args.config in ['2005', '2006', '2006_ori', '2007', '2008']:
    args.model_name = 'EfficientGCN-B2'
if args.config in ['2009', '2010', '2010_ori', '2011', '2012']:
    args.model_name = 'EfficientGCN-B4'

root_dir = f'./workdir/{args.config}_{args.model_name}_{args.benchmark}'
models = os.listdir(root_dir)

acc_dict = dict()
for model in models:
    model_path = os.path.join(root_dir, model)
    acc_path = os.path.join(model_path, 'reco_results.json')
    if not os.path.exists(acc_path):
        continue

    with open(acc_path, 'r') as f:
        acc = json.load(f)
        acc_top1 = float(acc.get('acc_top1'))
        acc_dict[model] = acc_top1

accs = np.array(list(acc_dict.values()))
acc_max = accs.max()
acc_mean = accs.mean()
acc_std = accs.std()
print(args.config, args.model_name, args.benchmark, len(accs), f'{acc_max:.2%}', f'{acc_mean:.2%}', f'{acc_std:.2%}')
print(sorted(acc_dict.items(), key=lambda v: (v[1], v[0]), reverse=True))
