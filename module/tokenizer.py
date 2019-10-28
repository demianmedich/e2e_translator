# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABCMeta
from abc import abstractmethod

from konlpy.tag import Mecab
from nltk.tokenize import word_tokenize


class Tokenizer(metaclass=ABCMeta):
    """Interface for tokenizer"""

    @abstractmethod
    def tokenize(self, sentence: str):
        """
        Tokenize sentence

        :param sentence: string to tokenize
        :return: Iterable list with tokenized string include <SOS>, <BOS>
        """
        ...

    @abstractmethod
    def tokenize_from_file(self, corpus_file_path: str):
        """
        Tokenize sentences from corpus file

        :param corpus_file_path: corpus file path
        :return: Iterable list with tokenized string list include <SOS>, <BOS>
        """
        ...


class MecabTokenizer(Tokenizer):
    """Mecab tokenizer. If you decide to use this, must install mecab from
    https://bitbucket.org/eunjeon/mecab-ko-dic/src/master/

    1. Install mecab-ko
    2. Install mecab-ko-dic
    """

    def __init__(self):
        super().__init__()
        self._impl = Mecab(dicpath='/usr/local/lib/mecab/dic/mecab-ko-dic')

    def tokenize(self, sentence: str):
        morphs = self._impl.morphs(sentence)
        return morphs

    def tokenize_from_file(self, corpus_file_path: str):
        morphs_list = []
        with open(corpus_file_path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = self._impl.morphs(line)
                morphs_list.append(line)
        return morphs_list


class NltkTokenizer(Tokenizer):
    """NLTK Tokenizer"""

    def tokenize(self, sentence: str):
        morphs = word_tokenize(sentence)
        return morphs

    def tokenize_from_file(self, corpus_file_path: str):
        morphs_list = []
        with open(corpus_file_path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = word_tokenize(line)
                morphs_list.append(line)
        return morphs_list
