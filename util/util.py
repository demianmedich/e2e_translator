# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from datetime import datetime

import torch
from torch import nn


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


def train_step(model: nn.Module,
               device: str,
               batch,
               enc_params: AttributeDict,
               dec_params: AttributeDict,
               optimizer,
               loss_func):
    src_seqs, src_lengths, tgt_seqs, tgt_lengths = batch
    src_seqs = src_seqs.to(device)
    src_lengths = src_lengths.to(device)
    tgt_seqs = tgt_seqs.to(device)
    tgt_lengths = tgt_lengths.to(device)
    logits, predictions = model(src_seqs, src_lengths, tgt_seqs, tgt_lengths)

    # To calculate loss, we should change shape of logits and labels
    # N is batch * seq_len, C is number of classes. (vocab size)
    # logits : (N by C)
    # labels : (N)
    logits = logits.contiguous().view(-1, dec_params.vocab_size)
    labels = tgt_seqs.contiguous().view(-1)

    loss = loss_func(logits, labels)

    # initialize buffer
    optimizer.zero_grad()

    # calculate gradient
    loss.backward()

    # update model parameter
    optimizer.step()

    return loss.item()


def eval_step(model: nn.Module,
              device: str,
              batch,
              enc_params: AttributeDict,
              dec_params: AttributeDict,
              loss_func):
    # TODO: need to implement
    src_seqs, src_lengths, tgt_seqs, tgt_lengths = batch
    src_seqs = src_seqs.to(device)
    src_lengths = src_lengths.to(device)
    tgt_seqs = tgt_seqs.to(device)
    tgt_lengths = tgt_lengths.to(device)

    model.eval()
    logits, predictions = model(src_seqs, src_lengths, tgt_seqs, tgt_lengths)

    # To calculate loss, we should change shape of logits and labels
    # N is batch * seq_len, C is number of classes. (vocab size)
    # logits : (N by C)
    # labels : (N)
    logits = logits.contiguous().view(-1, dec_params.vocab_size)
    labels = tgt_seqs.contiguous().view(-1)

    loss = loss_func(logits, labels)
    return loss.item()
