# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import runpy
import torch
from util import Estimator
from util import get_device

import torch.nn as nn

from util.tokens import PAD_TOKEN_ID


def parse_args():
    parser = argparse.ArgumentParser(
        description='Estimate model using separate params. please run on project root directory.')
    parser.add_argument('--params', required=True, type=str,
                        help='Search params from <project_root>/params/')
    parser.add_argument('--mode', required=True, type=str, help='One of [train, eval, inference]')
    parser.add_argument('--input', required=False, type=str, help='Inference input')
    return parser.parse_args()


def main():
    args = parse_args()
    global_dic = runpy.run_path(args.params)

    assert global_dic.get('common_params', None) is not None, \
        'common_params should be in params.py'
    assert global_dic.get('train_params', None) is not None, \
        'train_params should be in params.py'
    assert global_dic.get('eval_params', None) is not None, \
        'eval_params should be in params.py'
    assert args.mode in ['train', 'eval', 'inference'], \
        'mode should be one of [train, eval, inference]'

    common_params = global_dic['common_params']
    train_params = global_dic['train_params']
    eval_params = global_dic['eval_params']

    device = get_device()
    estimator = Estimator(device, common_params)
    loss_func = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    optimizer = torch.optim.Adam(estimator.get_model_parameters(), train_params.learning_rate)

    if args.mode == 'train':
        estimator.train(train_params, loss_func, optimizer)
    elif args.mode == 'eval':
        estimator.eval(eval_params, loss_func)
    else:
        assert args.input is not None, '--input required on inference mode.'
        estimator.inference(args.input, eval_params)


if __name__ == '__main__':
    main()
