# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.modules.transformer import TransformerDecoder as _TransformerDecoder
from torch.nn.modules.transformer import TransformerDecoderLayer
from torch.nn.modules.transformer import TransformerEncoder as _TransformerEncoder
from torch.nn.modules.transformer import TransformerEncoderLayer

from util import AttributeDict
from util.tokens import PAD_TOKEN_ID
from util.tokens import SOS_TOKEN_ID


class PositionalEncoding(nn.Module):
    """
    Positional encoding
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder
    https://pytorch.org/docs/stable/nn.html#transformer-layers
    """

    def __init__(self, params: AttributeDict):
        super().__init__()
        # mandatory
        self.d_model = params.d_model
        self.num_heads = params.num_heads
        self.vocab_size = params.vocab_size

        # optional
        self.num_layers = params.get('num_layers', 6)
        self.dim_feed_forward = params.get('dim_feed_forward', 2048)
        self.dropout_prob = params.get('dropout_prob', 0.1)
        self.pe_dropout_prob = params.get('pe_dropout_prob', 0.1)
        self.activation = params.get('activation', 'relu')
        self.max_seq_len = params.get('max_seq_len', 512)
        self.device = params.get('device', 'cpu')

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.pe_dropout_prob,
                                                      self.max_seq_len)
        encoder = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feed_forward,
            dropout=self.dropout_prob,
            activation=self.activation)
        # encoder will be cloned as much as num_layers
        norm = nn.LayerNorm(self.d_model)
        self.encoder_stack = _TransformerEncoder(encoder, self.num_layers, norm)
        self._init_parameter()

    def forward(self, src_seqs: torch.Tensor, src_lengths: torch.Tensor):
        """
        :param src_seqs: (batch_size, source_seq_len)
        :param src_lengths: (batch_size)
        :return: src_key_padding_mask, encoder_output: (source_seq_len, batch_size, d_model)
        """
        # source key padding mask to not focus padding
        # src_seqs: (batch_size, source_seq_len)
        # src_key_padding_mask: (batch_size, source_seq_len)
        src_key_padding_mask = src_seqs == PAD_TOKEN_ID

        embedding = self.embedding(src_seqs)
        embedding = self.positional_encoding(embedding)
        # embedding: (batch_size, source_seq_len, embedding_dim)

        # required src_input shape: (source_seq_len, batch_size, embedding_dim)
        embedding = torch.transpose(embedding, 0, 1)

        # [src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
        # that should be masked with float(‘-inf’) and False values will be unchanged.
        # This mask ensures that no information will be taken from position i if it is masked,
        # and has a separate mask for each sequence in a batch.
        return src_key_padding_mask, self.encoder_stack(embedding,
                                                        src_key_padding_mask=src_key_padding_mask)

    def init_embedding_weight(self, embedding_weight: np.ndarray):
        # Learning from model
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weight),
                                             requires_grad=False)

    def _init_parameter(self):
        """
        Initiate parameters in the transformer model.
        Just same as Transformer class implemented in py-torch.
        """
        for param in self.parameters():
            if param.dim() > 1:
                xavier_uniform_(param)


class TransformerDecoder(nn.Module):
    """
    Transformer decoder
    https://pytorch.org/docs/stable/nn.html#transformer-layers
    """

    def __init__(self, params: AttributeDict):
        super().__init__()
        # mandatory
        self.d_model = params.d_model
        self.num_heads = params.num_heads
        self.vocab_size = params.vocab_size

        # optional
        self.num_layers = params.get('num_layers', 6)
        self.dim_feed_forward = params.get('dim_feed_forward', 2048)
        self.dropout_prob = params.get('dropout_prob', 0.1)
        self.pe_dropout_prob = params.get('pe_dropout_prob', 0.1)
        self.activation = params.get('activation', 'relu')
        self.max_seq_len = params.get('max_seq_len', 512)
        self.device = params.get('device', 'cpu')

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.pe_dropout_prob,
                                                      self.max_seq_len)
        decoder_layer = TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feed_forward,
            dropout=self.dropout_prob,
            activation=self.activation
        )
        norm = nn.LayerNorm(self.d_model)
        # encoder will be cloned as much as num_layers
        self.decoder_stack = _TransformerDecoder(decoder_layer, self.num_layers, norm)
        self.linear_transform = nn.Linear(self.d_model, self.vocab_size)
        self._init_parameter()

    def _decode_step(self,
                     src_key_padding_mask: torch.Tensor,
                     enc_outputs: torch.Tensor,
                     tgt_seqs: torch.Tensor):
        # tgt_key_padding_mask: (batch_size, tgt_seq_length)
        # This makes output as NaN... why????????????
        # tgt_key_padding_mask = tgt_seqs == PAD_TOKEN_ID

        # memory_key_padding_mask: (batch_size, src_seq_length)
        memory_key_padding_mask = src_key_padding_mask

        # tgt_mask: (tgt_seq_len, tgt_seq_len)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seqs.size(-1))

        # required embedding shape: (tgt_seq_len, batch_size, embedding_dim)
        embedding = self.embedding(tgt_seqs)
        embedding = self.positional_encoding(embedding)
        embedding = torch.transpose(embedding, 0, 1)

        # output: (tgt_seq_len, batch_size, embedding_dim=n_model)
        output = self.decoder_stack(tgt=embedding,
                                    memory=enc_outputs,
                                    tgt_mask=tgt_mask,
                                    # tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)

        # output: (tgt_seq_len, batch_size, output_vocab_size)
        # transpose batch first
        output = self.linear_transform(output)
        output = torch.transpose(output, 0, 1)
        output = nn.functional.softmax(output, dim=-1)
        return output

    def forward(self,
                src_key_padding_mask: torch.Tensor,
                enc_outputs: torch.Tensor,
                tgt_seqs: torch.Tensor,
                tgt_lengths: torch.Tensor):
        if self.training:
            output = self._decode_step(src_key_padding_mask, enc_outputs, tgt_seqs)
        else:
            # evaluation or inference
            output = None
            batch_size = enc_outputs.size(1)
            ys = torch.ones(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_TOKEN_ID)
            for i in range(self.max_seq_len):
                output = self._decode_step(src_key_padding_mask, enc_outputs, ys)
                step_output = output[:, -1]
                _, top_index = step_output.data.topk(1)
                ys = torch.cat([ys, top_index], dim=1)

        return output

    def init_embedding_weight(self, embedding_weight: np.ndarray):
        # Learning from model
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weight),
                                             requires_grad=False)

    def _generate_square_subsequent_mask(self, size):
        """
        Note:
        [src/tgt/memory]_mask should be filled with float(‘-inf’) for the masked positions
        and float(0.0) else.
        These masks ensure that predictions for position i depend only on the unmasked positions j
        and are applied identically for each sequence in a batch.

        https://pytorch.org/docs/stable/nn.html#transformer-layers

        :param size: size of sequence
        :return: mask. masked positions filled with float('-inf'), float(0.0) others
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def _init_parameter(self):
        """
        Initiate parameters in the transformer model.
        Just same as Transformer class implemented in py-torch.
        """
        for param in self.parameters():
            if param.dim() > 1:
                xavier_uniform_(param)
