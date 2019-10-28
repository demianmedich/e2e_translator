# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from module import GruEncoder, GruDecoder
from module.tokenizer import NltkTokenizer
from util import AttributeDict

train_params = AttributeDict({
    "n_epochs": 5,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "src_tokenizer": NltkTokenizer,
    "tgt_tokenizer": NltkTokenizer,
    "src_vocab_filename": "src_vocab.txt",
    "src_word_embedding_filename": "src_word_embedding.npy",
    "tgt_vocab_filename": "tgt_vocab.txt",
    "tgt_word_embedding_filename": "tgt_word_embedding.npy",
    "src_corpus_filename": "korean-english-park.train.ko",
    "tgt_corpus_filename": "korean-english-park.train.en",
    "encoder": GruEncoder,
    "decoder": GruDecoder,
    "model_save_directory": "kor2eng-gru-gru"
})

eval_params = AttributeDict({
    "batch_size": 64,
    "src_tokenizer": NltkTokenizer,
    "tgt_tokenizer": NltkTokenizer,
    "src_vocab_filename": "src_vocab.txt",
    "src_word_embedding_filename": "src_word_embedding.npy",
    "tgt_vocab_filename": "tgt_vocab.txt",
    "tgt_word_embedding_filename": "tgt_word_embedding.npy",
    "src_corpus_filename": "korean-english-park.test.ko",
    "tgt_corpus_filename": "korean-english-park.test.en",
    "encoder": GruEncoder,
    "decoder": GruDecoder,
    "checkpoint_path": "kor2eng-gru-gru/2019-10-28-19-02-27-epoch_005/checkpoint.tar"
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
