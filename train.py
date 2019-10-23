# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from torch.utils.data import DataLoader
from torch import nn

from dataset import ParallelTextData
from dataset import collate_func
from module.embedding import make_fasttext_embedding_vocab_weight
from module.preprocess import MecabTokenizer
from module.preprocess import NltkTokenizer
from util import AttributeDict

config = AttributeDict({
    "n_epochs": 1,
    "batch_size": 64,
    "src_tokenizer": MecabTokenizer,
    "tgt_tokenizer": MecabTokenizer,
    "src_vocab_filename": "src_vocab.txt",
    "src_word_embedding_filename": "src_word_embedding.npy",
    "tgt_vocab_filename": "tgt_vocab.txt",
    "tgt_word_embedding_filename": "tgt_word_embedding.npy",
    "train_src_corpus_filename": "korean-english-park.train.ko",
    "train_tgt_corpus_filename": "korean-english-park.train.en",
    "embedding_dim": 100,
})


def check_config(config: AttributeDict):
    assert config.get('src_tokenizer', '') in [
        MecabTokenizer, NltkTokenizer
    ], 'src_tokenizer should be one of following [MecabTokenizer, NltkTokenizer]'
    assert config.get('tgt_tokenizer', '') in [
        MecabTokenizer, NltkTokenizer
    ], 'tgt_tokenizer should be one of following [MecabTokenizer, NltkTokenizer]'
    assert config.get('embedding_dim', None) is not None, \
        'embedding_dim should be given more than 0'
    assert config.get('src_vocab_filename', None) is not None, \
        'src_vocab_filename must not be None'
    assert config.get('tgt_vocab_filename', None) is not None, \
        'tgt_vocab_filename must not be None'
    assert config.get('src_word_embedding_filename', None) is not None, \
        'src_word_embedding_filename must not be None'
    assert config.get('tgt_word_embedding_filename', None) is not None, \
        'tgt_word_embedding_filename must not be None'
    assert config.get('train_src_corpus_filename', None) is not None, \
        'train_src_corpus_filename must not be None'
    assert config.get('train_tgt_corpus_filename', None) is not None, \
        'train_tgt_corpus_filename must not be None'


def ensure_vocab_embedding(
        tokenizer,
        vocab_file_path: str,
        word_embedding_file_path: str,
        corpus_file_path: str,
        embedding_dimen: int,
        tag: str,
):
    """
    :return: (word2id, id2word)
    """
    if not os.path.exists(vocab_file_path) or not os.path.exists(word_embedding_file_path):
        # Make source embedding
        print(f'{tag} embedding information is not exists.')

        embedding = make_fasttext_embedding_vocab_weight(
            tokenizer,
            corpus_file_path=corpus_file_path,
            vocab_path=vocab_file_path,
            weight_path=word_embedding_file_path,
            embedding_dim=embedding_dimen,
        )
        print(f'{tag} vocab size: {embedding.vocab_size}')

    with open(vocab_file_path, mode='r', encoding='utf-8') as f:
        tokens = f.readlines()
    word2id = {}
    id2word = {}
    for index, token in enumerate(tokens):
        token = token.strip()
        word2id[token] = index
        id2word[index] = token

    return word2id, id2word


# def train_model(model: nn.Module, )


if __name__ == '__main__':
    # 1. preprocessing
    tokenizer = MecabTokenizer()

    check_config(config)

    base_dir = os.getcwd()
    dataset_dir = os.path.join(base_dir, 'dataset')

    src_vocab_file_path = os.path.join(dataset_dir, config.src_vocab_filename)
    tgt_vocab_file_path = os.path.join(dataset_dir, config.tgt_vocab_filename)
    src_word_embedding_file_path = os.path.join(dataset_dir, config.src_word_embedding_filename)
    tgt_word_embedding_file_path = os.path.join(dataset_dir, config.tgt_word_embedding_filename)
    train_src_corpus_file_path = os.path.join(dataset_dir, config.train_src_corpus_filename)
    train_tgt_corpus_file_path = os.path.join(dataset_dir, config.train_tgt_corpus_filename)

    embedding_dim = config.embedding_dim

    src_word2id, src_id2word = ensure_vocab_embedding(
        tokenizer,
        src_vocab_file_path,
        src_word_embedding_file_path,
        train_src_corpus_file_path,
        embedding_dim,
        "Source")

    tgt_word2id, tgt_id2word = ensure_vocab_embedding(
        tokenizer,
        tgt_vocab_file_path,
        tgt_word_embedding_file_path,
        train_tgt_corpus_file_path,
        embedding_dim,
        "Target")

    # 2. train model
    dataset = ParallelTextData(tokenizer,
                               train_src_corpus_file_path,
                               train_tgt_corpus_file_path,
                               src_word2id,
                               tgt_word2id)
    data_loader = DataLoader(dataset,
                             batch_size=config.batch_size,
                             shuffle=True,
                             collate_fn=collate_func)

    for index, batch in enumerate(data_loader):
        # print(batch)
        if index == 0:
            src_seqs, src_seq_lengths, tgt_seqs, tgt_seq_lengths = batch[0], batch[1], batch[2], \
                                                                   batch[3]
            # [print(idx) for idx in src_seqs[0]]
            print([src_id2word[idx.item()] for idx in src_seqs[5]])
            print([tgt_id2word[idx.item()] for idx in tgt_seqs[5]])
        pass
