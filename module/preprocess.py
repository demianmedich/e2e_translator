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
        ...


class MecabTokenizer(Tokenizer):
    """Mecab tokenizer. If you decide to use this, must install mecab from
    https://bitbucket.org/eunjeon/mecab-ko-dic/src/master/

    1. Install mecab-ko
    2. Install mecab-ko-dic
    """

    def __init__(self,
                 dic_path='') -> None:
        super().__init__()
        self._impl = Mecab(dicpath='/usr/local/lib/mecab/dic/mecab-ko-dic')

    def tokenize(self, sentence: str):
        return self._impl.morphs(sentence)


class NltkTokenizer(Tokenizer):
    """NLTK Tokenizer"""

    def __init__(self) -> None:
        super().__init__()

    def tokenize(self, sentence: str):
        return word_tokenize(sentence)
