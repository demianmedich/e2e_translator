# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.optim import Adam

from module import Seq2Seq
from module import TransformerEncoder, TransformerDecoder
from module.tokenizer import NltkTokenizer, MecabTokenizer
from util import AttributeDict

common_params = AttributeDict({
    "model": Seq2Seq,
    "src_tokenizer": MecabTokenizer,
    "tgt_tokenizer": NltkTokenizer,
    "src_vocab_filename": "kor-mecab-fasttext",
    "tgt_vocab_filename": "eng-nltk-fasttext",
    "src_word_embedding_filename": "kor-mecab-fasttext-512d.npy",
    "tgt_word_embedding_filename": "eng-nltk-fasttext-512d.npy",
    "encoder_params": {
        "model": TransformerEncoder,
        "embedding_dim": 512,
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "dim_feed_forward": 2048,
        "pe_dropout_prob": 0.1,
        "dropout_prob": 0.1,
        "activation": "relu",
        "max_seq_len": 64,
    },
    "decoder_params": {
        "model": TransformerDecoder,
        "embedding_dim": 512,
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "dim_feed_forward": 2048,
        "pe_dropout_prob": 0.1,
        "dropout_prob": 0.1,
        "activation": "relu",
        "max_seq_len": 64,
    }
})

train_params = AttributeDict({
    "n_epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "optimizer": Adam,
    "betas": (0.9, 0.98),
    "eps": 1e-9,
    "src_corpus_filename": "korean-english-park.train.ko",
    "tgt_corpus_filename": "korean-english-park.train.en",
    "src_valid_corpus_filename": "korean-english-park.dev.ko",
    "tgt_valid_corpus_filename": "korean-english-park.dev.en",
    "model_save_directory": "kor2eng-trans-trans"
})

eval_params = AttributeDict({
    "batch_size": 128,
    "src_corpus_filename": "korean-english-park.test.ko",
    "tgt_corpus_filename": "korean-english-park.test.en",
    "checkpoint_path": "kor2eng-trans-trans/ckpt-19-11-01-01-05-39-epoch-020.tar"
})
