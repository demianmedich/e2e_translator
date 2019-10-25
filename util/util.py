# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch


class AttributeDict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, key, value):
        self[key] = value


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
