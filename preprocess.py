# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
from module.embedding import make_fasttext_embedding_vocab_weight
from module.embedding import make_word2vec_embedding_vocab_weight
from module.tokenizer import NltkTokenizer
from module.tokenizer import MecabTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Make vocab and word embedding. please run on project root directory.')
    parser.add_argument('--lang', required=True, help='kor or eng', type=str)
    parser.add_argument('--model', required=True, help='One of the [CBoW, Skipgram, FastText]',
                        type=str)
    parser.add_argument('--tokenizer', help='One of the [NLTK, Mecab]', type=str)
    parser.add_argument('--dim', required=True, help='Dimension of word embedding', type=int)
    parser.add_argument('--corpus_path', required=True, help='Corpus file path', type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    model = args.model.lower()
    tokenizer = args.tokenizer
    tokenizer_impl = None
    if tokenizer is not None:
        tokenizer = tokenizer.lower()
        assert tokenizer in ['nltk', 'mecab'], 'Only one of [nltk, mecab]'
        if tokenizer == 'nltk':
            tokenizer_impl = NltkTokenizer()
        else:
            tokenizer_impl = MecabTokenizer()
    embedding_dim = args.dim
    corpus_path = args.corpus_path

    assert model in ['cbow', 'skipgram', 'fasttext'], \
        'model should be one of [cbow, skipgram, fasttext]'
    assert embedding_dim is not None and embedding_dim > 0, \
        'embedding_dim should be bigger than 0'

    base_dir = os.getcwd()
    data_set_dir = os.path.join(base_dir, 'dataset')
    vocab_filename = f'{args.lang}-{tokenizer}-{model}'
    npy_filename = f'{args.lang}-{tokenizer}-{model}-{embedding_dim}d.npy'
    vocab_path = os.path.join(data_set_dir, vocab_filename)
    npy_path = os.path.join(data_set_dir, npy_filename)

    if model == 'cbow':
        embedding = make_word2vec_embedding_vocab_weight(tokenizer_impl, corpus_path, vocab_path,
                                                         npy_path, embedding_dim,
                                                         is_skip_gram=False)
    elif model == 'skipgram':
        embedding = make_word2vec_embedding_vocab_weight(tokenizer_impl, corpus_path, vocab_path,
                                                         npy_path, embedding_dim,
                                                         is_skip_gram=False)
    else:
        embedding = make_fasttext_embedding_vocab_weight(tokenizer_impl, corpus_path, vocab_path,
                                                         npy_path, embedding_dim)
    print(f'Embedding done', flush=True)
    print(f'  vocab size: {embedding.vocab_size}', flush=True)
    print(f'  saved vocab path: {vocab_path}', flush=True)
    print(f'  saved npy path: {npy_path}', flush=True)


if __name__ == '__main__':
    main()
