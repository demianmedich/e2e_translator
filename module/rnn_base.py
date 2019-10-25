# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from util.tokens import PAD_TOKEN_ID
from util.tokens import SOS_TOKEN_ID


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
        self.device = kwargs.get('device', 'cpu')

        self.embedding_lookup = nn.Embedding(self.vocab_size,
                                             self.embedding_dim,
                                             padding_idx=PAD_TOKEN_ID)
        self.rnn = nn.GRU(input_size=self.embedding_dim,
                          hidden_size=self.hidden_size,
                          batch_first=True,
                          num_layers=self.num_layers,
                          bidirectional=self.bidirectional,
                          dropout=self.dropout_prob)

    def forward(self, x, seq_lengths):
        embedding = self.embedding_lookup(x)
        packed_input = pack_padded_sequence(embedding, seq_lengths, batch_first=True)

        # If bidirectional is True,
        # output shape : (batch_size, seq_len, 2 * hidden_size)
        # hidden shape : (2 * num_layers, batch_size, hidden_size)
        output, hidden_state = self.rnn(packed_input)

        # output shape : (batch_size, seq_len, 2 * hidden_size)
        output, _ = pad_packed_sequence(output, batch_first=True, padding_value=PAD_TOKEN_ID)
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
            hidden_state = hidden_state[:self.num_layers] + hidden_state[self.num_layers:]

        # Standard rnn decoder cannot be bidirectional...
        return output, hidden_state

    def init_embedding_weight(self,
                              weight: np.ndarray):
        self.embedding_lookup.weight = nn.Parameter(torch.from_numpy(weight), requires_grad=False)


class GruDecoder(nn.Module):
    """Gru Decoder"""

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = kwargs.get('num_layers', 1)
        self.dropout_prob = kwargs.get('dropout_prob', 0.0)
        self.device = kwargs.get('device', 'cpu')

        self.embedding_lookup = nn.Embedding(self.vocab_size,
                                             self.embedding_dim,
                                             padding_idx=PAD_TOKEN_ID)
        self.rnn = nn.GRU(input_size=self.embedding_dim,
                          hidden_size=self.hidden_size,
                          batch_first=True,
                          bidirectional=False,
                          num_layers=self.num_layers,
                          dropout=self.dropout_prob)
        self.linear_transform = nn.Linear(self.hidden_size, self.vocab_size)
        self.decoder_output_func = nn.functional.log_softmax

    def forward(self, encoder_output, encoder_hidden_state, tgt_seqs, tgt_seq_lengths):
        # Decoder GRU cannot be bidirectional.
        # encoder output:  (batch, seq_len, num_directions * hidden_size) => batch_first=True
        # encoder hidden:  (num_layers * num_directions, batch, hidden_size)

        batch_size = encoder_output.size(0)
        max_seq_len = tgt_seqs.size(-1)
        # (Batch_size)
        initial_input = batch_size * [SOS_TOKEN_ID]
        initial_input = torch.tensor(initial_input, dtype=torch.long, device=self.device).unsqueeze(
            -1)

        # predicted output will be saved here
        logits = torch.zeros(max_seq_len, batch_size, self.vocab_size, device=self.device)

        decoder_input = initial_input
        prev_hidden_state = encoder_hidden_state

        predictions = []
        for t in range(tgt_seqs.size(-1)):
            decoder_output, hidden_state = self.step(t, decoder_input, prev_hidden_state)
            logits[t] = decoder_output

            if self.training:
                decoder_input = tgt_seqs[:, t]
            else:
                # Greedy search
                top_value, top_index = decoder_output.data.topk(1)
                decoder_input = top_index.squeeze(-1).detach()
                predictions.append(decoder_input.cpu())

            decoder_input = decoder_input.long().unsqueeze(-1)
            prev_hidden_state = hidden_state

        # To calculate loss, we should change shape of logits and labels
        # N is batch * seq_len, C is number of classes. (vocab size)
        # logits : (N by C)
        # labels : (N)
        logits = logits.transpose(0, 1)
        logits = logits.contiguous().view(-1, self.vocab_size)
        labels = tgt_seqs.contiguous().view(-1)

        return logits, labels, predictions

    def step(self, t, inputs, prev_hidden_state):
        embedding = self.embedding_lookup(inputs)

        outputs, hidden_state = self.rnn(embedding, prev_hidden_state)
        outputs = self.linear_transform(outputs.transpose(0, 1).squeeze(0))

        if self.decoder_output_func:
            outputs = self.decoder_output_func(outputs, dim=-1)

        # To save in logits, seq_len should be removed.
        return outputs, hidden_state

    def init_embedding_weight(self,
                              weight: np.ndarray):
        self.embedding_lookup.weight = nn.Parameter(torch.from_numpy(weight), requires_grad=False)
