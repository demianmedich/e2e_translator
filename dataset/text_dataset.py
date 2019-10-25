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
                 src_tokenizer,
                 tgt_tokenizer,
                 src_corpus_path: str,
                 tgt_corpus_path: str,
                 max_src_length: int,
                 max_tgt_length: int,
                 src_word2id: dict,
                 tgt_word2id: dict):
        super().__init__()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_word2id = src_word2id
        self.tgt_word2id = tgt_word2id
        self.src_max_length = max_src_length
        self.tgt_max_length = max_tgt_length

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
        src_tokens = []
        tgt_tokens = []

        for i, token in enumerate(self.src_tokenizer.tokenize(src)):
            if i == self.src_max_length:
                break
            if token in self.src_word2id:
                src_tokens.append(self.src_word2id[token])
            else:
                src_tokens.append(self.src_word2id[UNK_TOKEN])
        src_tokens.append(self.src_word2id[EOS_TOKEN])

        for i, token in enumerate(self.tgt_tokenizer.tokenize(tgt)):
            if i == self.tgt_max_length:
                break
            if token in self.tgt_word2id:
                tgt_tokens.append(self.tgt_word2id[token])
            else:
                tgt_tokens.append(self.tgt_word2id[UNK_TOKEN])
        tgt_tokens.append(self.tgt_word2id[EOS_TOKEN])

        return src_tokens, tgt_tokens

    def __len__(self):
        return len(self.pair_sentences)

    @staticmethod
    def pad_tensor(sentence, max_len, pad_value=0):
        """Append padding to one sentence"""
        pad_size = max_len - len(sentence)
        for _ in range(pad_size):
            sentence.append(pad_value)

    @staticmethod
    def pad_tokenized_sequence(tokenized_sequence, max_length=None):
        """
        Append padding to several sentences

        :param tokenized_sequence: sequence with several tokens.
        :param max_length: maximum length for tokenized sequence.
        :return: padded_token_sequence, sequence_length excluding padding
        """
        sequence_lengths = torch.tensor(
            [len(sentence) for sentence in tokenized_sequence])
        if max_length is None:
            max_length = int(max(sequence_lengths).item())

        [ParallelTextData.pad_tensor(sentence, max_length,
                                     pad_value=SPECIAL_TOKENS.index(PAD_TOKEN)) for sentence in
         tokenized_sequence]
        padded_tokens = torch.tensor(tokenized_sequence)
        return padded_tokens, sequence_lengths

    def collate_func(self, batch):
        """
        Called whenever mini-batch decided.
        :returns src_seqs, src_seq_lengths, tgt_seqs, tgt_seq_lengths
        """
        src_sequences, src_sequence_lengths = ParallelTextData.pad_tokenized_sequence(
            [src for src, _ in batch], self.src_max_length)
        tgt_sequences, tgt_sequence_lengths = ParallelTextData.pad_tokenized_sequence(
            [tgt for _, tgt in batch], self.tgt_max_length)
        src_sequence_lengths, sorted_idx = src_sequence_lengths.sort(descending=True)

        src_sequences = src_sequences[sorted_idx].contiguous()
        tgt_sequences = tgt_sequences[sorted_idx].contiguous()
        tgt_sequence_lengths = tgt_sequence_lengths[sorted_idx].contiguous()

        return src_sequences.long(), src_sequence_lengths.int(), \
               tgt_sequences.long(), tgt_sequence_lengths.int()
