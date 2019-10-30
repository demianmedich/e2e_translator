# coding: utf-8
# DEPRECATED!!!!!
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import nltk
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ParallelTextDataSet
from module import Seq2Seq
from module.tokenizer import MecabTokenizer
from module.tokenizer import NltkTokenizer
from params.params import decoder_params
from params.params import encoder_params
from params.params import eval_params
from util import AttributeDict
from util import eval_step
from util import get_device
from util.tokens import PAD_TOKEN_ID
from util import index2word


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


def eval_model(model: nn.Module,
               loss_func,
               test_data_loader: DataLoader,
               device: str,
               id2word: dict):
    model.eval()

    with torch.no_grad():
        losses = []
        data_length = len(test_data_loader)

        predictions = []
        target_sequences = []

        with tqdm(test_data_loader, total=data_length, desc=f'EVAL') as tqdm_iterator:
            for _, batch in enumerate(tqdm_iterator):
                _, _, tgt_seqs, tgt_lengths = batch

                # TODO: PAD 고려??
                loss, logits, predicted_sentences = eval_step(model, device, batch, loss_func)
                # preds = torch.cat(preds).view(-1, len(preds))

                for predicted_sentence in predicted_sentences:
                    predictions.append(index2word(id2word, predicted_sentence))

                for tgt_seq in tgt_seqs:
                    target_sequences.append(index2word(id2word, tgt_seq))

                losses.append(loss)
                tqdm_iterator.set_postfix_str(f'loss: {loss:05.3f}')

        bleu_score = nltk.translate.bleu_score.corpus_bleu(target_sequences, predictions)

    avg_loss = np.mean(losses)
    return avg_loss, bleu_score


def main():
    check_params(eval_params)
    device = get_device()
    print(f'  Available device is {device}')

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
    src_corpus_file_path = os.path.join(dataset_dir, eval_params.src_corpus_filename)
    tgt_corpus_file_path = os.path.join(dataset_dir, eval_params.tgt_corpus_filename)

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
    # encoder.init_embedding_weight(src_embedding)

    decoder_params.vocab_size = len(tgt_word2id)
    decoder_params.device = device
    decoder = eval_params.decoder(decoder_params)
    # decoder.init_embedding_weight(tgt_embedding)

    model: nn.Module = Seq2Seq(encoder, decoder)
    loss_func = nn.CrossEntropyLoss()

    checkpoint = torch.load(os.path.join(base_dir, checkpoint_path))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    dataset = ParallelTextDataSet(
        src_tokenizer,
        tgt_tokenizer,
        src_corpus_file_path,
        tgt_corpus_file_path,
        encoder_params.max_seq_len,
        decoder_params.max_seq_len,
        src_word2id,
        tgt_word2id
    )
    data_loader = DataLoader(dataset,
                             eval_params.batch_size,
                             collate_fn=dataset.collate_func)

    # avg_loss, bleu_score = eval_model(model, loss_func, data_loader, device, tgt_id2word)
    avg_loss, bleu_score = eval_model(model, loss_func, data_loader, device, tgt_id2word)


if __name__ == '__main__':
    print('***** Eval start *****')
    main()
    print('***** Eval end *****')
