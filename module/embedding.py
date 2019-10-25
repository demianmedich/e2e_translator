# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from gensim.models import FastText
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.fasttext import load_facebook_model

from util.tokens import SPECIAL_TOKENS


class WordEmbedding(metaclass=ABCMeta):
    """Interface for Word embedding"""

    @abstractmethod
    def word2vec(self, words):
        ...

    @abstractmethod
    def vec2word(self, vectors):
        ...

    @abstractmethod
    def get_embedding_matrix(self) -> np.ndarray:
        ...


class Word2VecEmbedding(WordEmbedding):

    def __init__(self,
                 is_skip_gram=True,
                 sentences=None,
                 corpus_file_path: str = None,
                 dim: int = 100,
                 saved_model_path: str = None) -> None:
        """Constructor for Word2Vec classes (RAII)"""
        super().__init__()

        if sentences is not None:
            self._impl = Word2Vec(sg=int(is_skip_gram), size=dim, sentences=sentences)
        elif corpus_file_path is not None:
            self._impl = Word2Vec(sg=int(is_skip_gram), size=dim, corpus_file=corpus_file_path)
        elif saved_model_path is not None:
            # load from saved FastText embedding file
            self._impl = KeyedVectors.load_word2vec_format(saved_model_path)
        else:
            raise AssertionError(
                'sentences or corpus_file_path should be given as not None')

        self.vocab = self._impl.wv.index2word
        self.vocab_size = len(self.vocab)
        self.embedding_matrix = []
        for word in self.vocab:
            self.embedding_matrix.append(self._impl.wv[word])

    def word2vec(self, words):
        return [self._impl.wv.get_vector(w) for w in words]

    def vec2word(self, vectors):
        return [self._impl.similar_by_vector(v, topn=1) for v in vectors]

    def get_embedding_matrix(self) -> np.ndarray:
        return np.array(self.embedding_matrix)


class FastTextEmbedding(WordEmbedding):
    """FastText word embedding"""

    def __init__(self,
                 sentences=None,
                 corpus_file_path: str = None,
                 dim: int = 100,
                 saved_model_path: str = None,
                 save_path=None) -> None:
        """Constructor for FastTextEmbedding classes (RAII)"""
        super().__init__()

        if sentences is not None:
            self._impl = FastText(size=dim, sentences=sentences)
        elif corpus_file_path is not None:
            self._impl = FastText(size=dim, corpus_file=corpus_file_path)
        elif saved_model_path is not None:
            # load from saved FastText embedding file
            self._impl = load_facebook_model(saved_model_path)
        else:
            raise AssertionError(
                'sentences or corpus_file_path should be given as not None')

        self.vocab = self._impl.wv.index2word
        self.vocab_size = len(self.vocab)
        self.embedding_matrix = []
        for word in self.vocab:
            self.embedding_matrix.append(self._impl.wv[word])

    def word2vec(self, words):
        return [self._impl.wv.get_vector(w) for w in words]

    def vec2word(self, vectors):
        return [self._impl.similar_by_vector(v, topn=1)[0] for v in vectors]

    def get_embedding_matrix(self) -> np.ndarray:
        return np.array(self.embedding_matrix)


def make_word2vec_embedding_vocab_weight(
        tokenizer,
        corpus_file_path: str,
        vocab_path: str,
        weight_path: str,
        embedding_dim):
    if os.path.exists(vocab_path):
        os.remove(vocab_path)
    if os.path.exists(weight_path):
        os.remove(weight_path)

    sentences = tokenizer.tokenize_from_file(corpus_file_path=corpus_file_path)
    embedding = Word2VecEmbedding(sentences=sentences, dim=embedding_dim)
    vocab = [f'{token}\n' for token in embedding.vocab]
    with open(vocab_path, mode='w', encoding='utf-8') as f:
        for token in SPECIAL_TOKENS:
            f.write(token + '\n')
        f.writelines(vocab)
    matrix = embedding.get_embedding_matrix()
    for index, token in enumerate(SPECIAL_TOKENS):
        matrix = np.insert(matrix, index, np.random.randn(embedding_dim), axis=0)
    np.save(weight_path, matrix)
    return embedding


def make_fasttext_embedding_vocab_weight(
        tokenizer,
        corpus_file_path: str,
        vocab_path: str,
        weight_path: str,
        embedding_dim):
    if os.path.exists(vocab_path):
        os.remove(vocab_path)
    if os.path.exists(weight_path):
        os.remove(weight_path)

    sentences = tokenizer.tokenize_from_file(corpus_file_path=corpus_file_path)
    embedding = FastTextEmbedding(sentences=sentences, dim=embedding_dim)
    vocab = [f'{token}\n' for token in embedding.vocab]
    vocab_size_with_special_tokens = embedding.vocab_size + len(SPECIAL_TOKENS)

    with open(vocab_path, mode='w', encoding='utf-8') as f:
        for token in SPECIAL_TOKENS:
            f.write(token + '\n')
        f.writelines(vocab)
    matrix = embedding.get_embedding_matrix()
    for index, token in enumerate(SPECIAL_TOKENS):
        matrix = np.insert(matrix, index, np.random.randn(embedding_dim), axis=0)
    np.save(weight_path, matrix)
    return embedding
