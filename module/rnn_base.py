# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
from torch import nn

from util import AttributeDict
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class GruEncoder(nn.Module):
    """Gru Encoder"""

    def __init__(self,
                 config: AttributeDict):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_size
        self.hidden_size = config.hidden_size
        self.bidirectional = config.get('bidirectional', False)
        self.num_layers = config.get('num_layers', 1)
        self.dropout_prob = config.get('dropout_prob', 0.0)

        self.embedding_lookup = nn.Embedding(self.vocab_size,
                                             self.embedding_dim,
                                             padding_idx=0)
        self.rnn = nn.GRU(self.embedding_dim,
                          self.hidden_size,
                          batch_first=True,
                          num_layers=self.num_layers,
                          bidirectional=self.bidirectional)

    def forward(self, x, seq_lengths):
        x = self.embedding_lookup(x)
        # x = pack_padded_sequence(x, seq_lengths, )

    def init_embedding_weight(self,
                              weight: np.ndarray):
        self.embedding_lookup.weight = nn.Parameter(torch.from_numpy(weight),
                                                    requires_grad=False)
