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
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.bidirectional = kwargs.get('bidirectional', False)
        self.num_layers = kwargs.get('num_layers', 1)
        self.dropout_prob = kwargs.get('dropout_prob', 0.0)

        self.embedding_lookup = nn.Embedding(self.vocab_size,
                                             self.embedding_dim,
                                             padding_idx=0)
        self.rnn = nn.GRU(self.embedding_dim,
                          self.hidden_size,
                          batch_first=True,
                          num_layers=self.num_layers,
                          bidirectional=self.bidirectional,
                          dropout=self.dropout_prob)

    def forward(self, x, seq_lengths):
        x = self.embedding_lookup(x)
        packed_input = pack_padded_sequence(x, seq_lengths, batch_first=True)
        init_hidden_state = torch.zeros(
            self.num_layers * 2 if self.bidirectional else 1,
            x.size(0),
            self.hidden_size
        )
        _, hidden_state = self.rnn(packed_input, init_hidden_state)
        return hidden_state

    def init_embedding_weight(self,
                              weight: np.ndarray):
        self.embedding_lookup.weight = nn.Parameter(torch.from_numpy(weight),
                                                    requires_grad=False)
