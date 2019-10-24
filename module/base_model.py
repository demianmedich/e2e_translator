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

    def forward(self, src_seqs, src_seq_lengths, tgt_seqs, tgt_seq_lengths):
        encoded_result = self.encoder(src_seqs, src_seq_lengths)
        return self.decoder(*encoded_result, tgt_seqs, tgt_seq_lengths)
