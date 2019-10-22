# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
from torch import nn

from util import AttributeDict


# class Seq2Seq(nn.Module):
#     def __init__(self):


class TranslatorRnn(nn.Module):

    def __init__(self,
                 config: AttributeDict,
                 ) -> None:
        super().__init__()

        vocab_size = config.vocab_size
        embedding_size = config.embedding_size

        self.embedding_lookup = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        # self.

    def init_embedding_weight(self,
                              weight: np.ndarray):
        self.embedding_lookup.weight = nn.Parameter(torch.from_numpy(weight), requires_grad=False)

    def forward(self, x, seq_lengths):
        pass
