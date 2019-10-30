# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from datetime import datetime

import torch
from torch import nn
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


def eval_step(model: nn.Module,
              device: str,
              batch,
              loss_func):
    model.eval()
    src_seqs, src_lengths, tgt_seqs, tgt_lengths = batch
    src_seqs = src_seqs.to(device)
    src_lengths = src_lengths.to(device)
    tgt_seqs = tgt_seqs.to(device)
    tgt_lengths = tgt_lengths.to(device)

    # logits: (batch_size, max_seq_len, tgt_vocab_size)
    logits = model(src_seqs, src_lengths, tgt_seqs, tgt_lengths)
    _, indices = torch.max(logits.detach().cpu(), dim=-1, keepdim=True)
    predictions = indices.squeeze(-1)

    # To calculate loss, we should change shape of logits and labels
    # N is batch * seq_len, C is number of classes. (vocab size)
    # logits : (N by C)
    # labels : (N)
    # TODO: PAD 고려??
    logits = logits.contiguous().view(-1, logits.size(-1))
    labels = tgt_seqs.contiguous().view(-1)

    loss = loss_func(logits, labels)
    return loss.item(), logits, predictions


def train_step(model: nn.Module,
               device: str,
               batch,
               optimizer,
               loss_func):
    model.train()
    src_seqs, src_lengths, tgt_seqs, tgt_lengths = batch
    src_seqs = src_seqs.to(device)
    src_lengths = src_lengths.to(device)
    tgt_seqs = tgt_seqs.to(device)
    tgt_lengths = tgt_lengths.to(device)
    logits = model(src_seqs, src_lengths, tgt_seqs, tgt_lengths)

    # To calculate loss, we should change shape of logits and labels
    # N is batch * seq_len, C is number of classes. (vocab size)
    # logits : (N by C)
    # labels : (N)
    logits = logits.contiguous().view(-1, logits.size(-1))
    labels = tgt_seqs.contiguous().view(-1)
    indices_except_padding = [labels != 0]
    loss = loss_func(logits[indices_except_padding], labels[indices_except_padding])

    # initialize buffer
    optimizer.zero_grad()

    # calculate gradient
    loss.backward()

    # update model parameter
    optimizer.step()

    return loss.item()


def index2word(tgt_id2word: dict, tokens: torch.Tensor) -> list:
    sentence = []
    for token in tokens:
        token = token.item()
        if token == PAD_TOKEN_ID:
            break
        if token in tgt_id2word:
            sentence.append(tgt_id2word[token])
    return sentence
