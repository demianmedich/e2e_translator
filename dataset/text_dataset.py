# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.utils import data

from util.tokens import EOS_TOKEN
from util.tokens import PAD_TOKEN
from util.tokens import SPECIAL_TOKENS
from util.tokens import UNK_TOKEN


class ParallelTextData(data.Dataset):

    def __init__(self,
                 tokenizer,
                 src_corpus_path: str,
                 tgt_corpus_path: str,
                 src_word2id: dict,
                 tgt_word2id: dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.src_word2id = src_word2id
        self.tgt_word2id = tgt_word2id

        with open(src_corpus_path, mode='r', encoding='utf-8') as src, \
                open(tgt_corpus_path, mode='r', encoding='utf-8') as tgt:
            self.pair_sentences = []

            while True:
                src_line = src.readline().strip()
                if not src_line:
                    break

                tgt_line = tgt.readline().strip()
                if not tgt_line:
                    break

                self.pair_sentences.append((src_line, tgt_line))

    def __getitem__(self, index: int):
        src, tgt = self.pair_sentences[index]
        # print(src)
        # print(tgt)
        src_tokens = []
        tgt_tokens = []

        for token in self.tokenizer.tokenize(src):
            if token in self.src_word2id:
                src_tokens.append(self.src_word2id[token])
            else:
                src_tokens.append(self.src_word2id[UNK_TOKEN])
        src_tokens.append(self.src_word2id[EOS_TOKEN])

        for token in self.tokenizer.tokenize(tgt):
            if token in self.tgt_word2id:
                tgt_tokens.append(self.tgt_word2id[token])
            else:
                tgt_tokens.append(self.tgt_word2id[UNK_TOKEN])
        tgt_tokens.append(self.tgt_word2id[EOS_TOKEN])

        return src_tokens, tgt_tokens

    def __len__(self):
        return len(self.pair_sentences)


def pad_tensor(sentence, max_len, pad_value=0):
    """Append padding to one sentence"""
    pad_size = max_len - len(sentence)
    for _ in range(pad_size):
        sentence.append(pad_value)


def pad_tokenized_sequence(tokenized_sequence):
    """
    Append padding to several sentences

    :param tokenized_sequence: sequence with several tokens
    :return: padded_token_sequence, sequence_length excluding padding
    """
    sequence_lengths = torch.tensor(
        [len(sentence) for sentence in tokenized_sequence])
    max_len = int(max(sequence_lengths).item())
    [pad_tensor(sentence, max_len=max_len,
                pad_value=SPECIAL_TOKENS.index(PAD_TOKEN)) for sentence in
     tokenized_sequence]
    padded_tokens = torch.tensor(tokenized_sequence)
    return padded_tokens, sequence_lengths


def collate_func(batch):
    """
    Called whenever mini-batch decided.
    :returns src_seqs, src_seq_lengths, tgt_seqs, tgt_seq_lengths
    """
    src_sequences, src_sequence_lengths = pad_tokenized_sequence(
        [src for src, _ in batch])
    tgt_sequences, tgt_sequence_lengths = pad_tokenized_sequence(
        [tgt for _, tgt in batch])
    src_sequence_lengths, sorted_idx = src_sequence_lengths.sort(
        descending=True)

    src_sequences = src_sequences[sorted_idx].contiguous()
    tgt_sequences = tgt_sequences[sorted_idx].contiguous()

    return src_sequences.long(), src_sequence_lengths.int(), \
           tgt_sequences.long(), tgt_sequence_lengths.int()
