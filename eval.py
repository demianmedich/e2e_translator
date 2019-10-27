# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from util import AttributeDict
from module.preprocess import NltkTokenizer
from module.preprocess import MecabTokenizer

eval_params = AttributeDict({
    "src_tokenizer": NltkTokenizer,
    "tgt_tokenizer": NltkTokenizer,
    "src_vocab_filename": "src_vocab.txt",
    "src_word_embedding_filename": "src_word_embedding.npy",
    "tgt_vocab_filename": "tgt_vocab.txt",
    "tgt_word_embedding_filename": "tgt_word_embedding.npy",
    "src_corpus_filename": "korean-english-park.dev.ko",
    "tgt_corpus_filename": "korean-english-park.dev.en",
    "model_save_directory": "kor2eng-gru-gru"
})

encoder_params = AttributeDict({
    "embedding_dim": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "dropout_prob": 0.3,
    "bidirectional": True,
    "max_seq_len": 100,
})

decoder_params = AttributeDict({
    "embedding_dim": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "dropout_prob": 0.3,
    "max_seq_len": 100,
    "beam_size": 3,
})


def check_params(config: AttributeDict):
    assert config.get('src_tokenizer', '') in [
        MecabTokenizer, NltkTokenizer
    ], 'src_tokenizer should be one of following [MecabTokenizer, NltkTokenizer]'
    assert config.get('tgt_tokenizer', '') in [
        MecabTokenizer, NltkTokenizer
    ], 'tgt_tokenizer should be one of following [MecabTokenizer, NltkTokenizer]'
    assert config.get('src_vocab_filename', None) is not None, \
        'src_vocab_filename must not be None'
    assert config.get('tgt_vocab_filename', None) is not None, \
        'tgt_vocab_filename must not be None'
    assert config.get('src_word_embedding_filename', None) is not None, \
        'src_word_embedding_filename must not be None'
    assert config.get('tgt_word_embedding_filename', None) is not None, \
        'tgt_word_embedding_filename must not be None'
    assert config.get('src_corpus_filename', None) is not None, \
        'src_corpus_filename must not be None'
    assert config.get('tgt_corpus_filename', None) is not None, \
        'tgt_corpus_filename must not be None'


def main():
    check_params(eval_params)
    torch.load()


if __name__ == '__main__':
    main()
