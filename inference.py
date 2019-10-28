# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
import torch
from torch import nn

from module import Seq2Seq
from module.tokenizer import MecabTokenizer
from module.tokenizer import NltkTokenizer
from params import decoder_params
from params import encoder_params
from params import eval_params
from util import AttributeDict
from util.tokens import PAD_TOKEN_ID
from util.tokens import UNK_TOKEN_ID


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
    assert config.get('encoder', None) is not None, \
        'encoder should not be None'
    assert config.get('decoder', None) is not None, \
        'decoder should not be None'
    assert config.get('checkpoint_path', None) is not None, \
        'model_path should not be None'


def check_vocab_embedding(
        vocab_file_path: str,
        word_embedding_file_path: str,
):
    """
    :return: word2id, id2word, embedding_matrix
    """

    with open(vocab_file_path, mode='r', encoding='utf-8') as f:
        tokens = f.readlines()
    word2id = {}
    id2word = {}
    for index, token in enumerate(tokens):
        token = token.strip()
        if len(token) == 0:
            continue
        word2id[token] = index
        id2word[index] = token

    embedding_matrix = np.load(word_embedding_file_path)

    return word2id, id2word, embedding_matrix


def pad_token(sentence, max_len, pad_value=PAD_TOKEN_ID):
    """Append padding to one sentence"""
    pad_size = max_len - len(sentence)
    for _ in range(pad_size):
        sentence.append(pad_value)


def main():
    check_params(eval_params)
    device = 'cpu'

    src_tokenizer = eval_params.src_tokenizer()
    tgt_tokenizer = eval_params.tgt_tokenizer()
    checkpoint_path = eval_params.checkpoint_path

    base_dir = os.getcwd()
    dataset_dir = os.path.join(base_dir, 'dataset')

    src_vocab_file_path = os.path.join(dataset_dir, eval_params.src_vocab_filename)
    tgt_vocab_file_path = os.path.join(dataset_dir, eval_params.tgt_vocab_filename)
    src_word_embedding_file_path = os.path.join(dataset_dir,
                                                eval_params.src_word_embedding_filename)
    tgt_word_embedding_file_path = os.path.join(dataset_dir,
                                                eval_params.tgt_word_embedding_filename)

    src_word2id, src_id2word, src_embedding = check_vocab_embedding(
        src_vocab_file_path,
        src_word_embedding_file_path
    )
    tgt_word2id, tgt_id2word, tgt_embedding = check_vocab_embedding(
        tgt_vocab_file_path,
        tgt_word_embedding_file_path
    )

    encoder_params.vocab_size = len(src_word2id)
    encoder_params.device = device
    encoder = eval_params.encoder(encoder_params)

    decoder_params.vocab_size = len(tgt_word2id)
    decoder_params.device = device
    decoder = eval_params.decoder(decoder_params)

    model: nn.Module = Seq2Seq(encoder, decoder)

    checkpoint = torch.load(os.path.join(base_dir, checkpoint_path))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    src_max_length = encoder_params.max_seq_len
    src_seqs = '저장된 검색어가 없습니다.'
    print(f'Input sequence: {src_seqs}')
    with torch.no_grad():
        src_tokenized = src_tokenizer.tokenize(src_seqs)
        print(src_tokenized)

        temp_tokenized = []
        for word in src_tokenized:
            if word in src_word2id:
                temp_tokenized.append(src_word2id[word])
            else:
                temp_tokenized.append(UNK_TOKEN_ID)
        src_tokenized = temp_tokenized

        pad_token(src_tokenized, src_max_length)
        print(src_tokenized)

        src_padded_tokens = torch.tensor(src_tokenized, dtype=torch.long, device=device).unsqueeze(
            0)
        src_length = torch.tensor(len(src_tokenized)).unsqueeze(0)
        logits, preds = model(src_padded_tokens, src_length, None, None)

        sentence = []
        for token in preds:
            token = token.item()
            if token == PAD_TOKEN_ID:
                break
            sentence.append(tgt_id2word[token].strip())
        print(sentence)
        print(len(sentence))


if __name__ == '__main__':
    main()
