# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from datetime import datetime

import torch

from util.tokens import SOS_TOKEN_ID
from util.tokens import EOS_TOKEN_ID
from util.tokens import PAD_TOKEN_ID


class AttributeDict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, key, value):
        self[key] = value


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_checkpoint_dir_path(epoch: int) -> str:
    date_fmt = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    return f'{date_fmt}-epoch_{epoch:03d}'


def pad_token(tokens, max_len, pad_value=PAD_TOKEN_ID):
    """Append padding to one sentence"""
    pad_size = max_len - len(tokens)
    for _ in range(pad_size):
        tokens.append(pad_value)


def index2word(tgt_id2word: dict, tokens: torch.Tensor) -> list:
    sentence = []
    for token in tokens:
        token = token.item()
        if token == SOS_TOKEN_ID:
            continue
        if token == EOS_TOKEN_ID:
            break
        if token == PAD_TOKEN_ID:
            break
        if token in tgt_id2word:
            sentence.append(tgt_id2word[token])
    return sentence
