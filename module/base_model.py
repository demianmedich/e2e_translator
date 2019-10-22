# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch import nn


class Seq2Seq(nn.Module):

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, seq_lengths):
        return self.decoder(self.encoder(x, seq_lengths))
