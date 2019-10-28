# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ParallelTextDataSet
from module import Seq2Seq
from module.embedding import make_fasttext_embedding_vocab_weight
from module.tokenizer import MecabTokenizer
from module.tokenizer import NltkTokenizer
from params import decoder_params
from params import encoder_params
from params import train_params
from util import AttributeDict
from util import get_checkpoint_dir_path
from util import get_device
from util import train_step


def check_params(config: AttributeDict):
    assert isinstance(config.get('learning_rate'), float), \
        'learning_rate should be float value.'
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


def train_model(model: nn.Module,
                optimizer,
                loss_func,
                data_loader: DataLoader,
                device: str,
                train_params: AttributeDict,
                enc_params: AttributeDict,
                dec_params: AttributeDict,
                epoch: int):
    # Set train flag
    model.train()
    n_epochs = train_params.n_epochs
    losses = []
    data_length = len(data_loader)

    with tqdm(data_loader, total=data_length, desc=f'Epoch {epoch:03d}') as tqdm_iterator:
        for _, batch in enumerate(tqdm_iterator):
            loss = train_step(model, device, batch, optimizer, loss_func)
            losses.append(loss)
            tqdm_iterator.set_postfix_str(f'loss: {loss:05.3f}')

    avg_loss = np.mean(losses)
    print(f'Epochs [{epoch}/{n_epochs}] avg losses: {avg_loss:05.3f}')
    return avg_loss


def main():
    check_params(train_params)

    device = get_device()
    print(f'  Available device is {device}')

    src_tokenizer = train_params.src_tokenizer()
    tgt_tokenizer = train_params.tgt_tokenizer()

    base_dir = os.getcwd()
    dataset_dir = os.path.join(base_dir, 'dataset')

    src_vocab_file_path = os.path.join(dataset_dir, train_params.src_vocab_filename)
    tgt_vocab_file_path = os.path.join(dataset_dir, train_params.tgt_vocab_filename)
    src_word_embedding_file_path = os.path.join(dataset_dir,
                                                train_params.src_word_embedding_filename)
    tgt_word_embedding_file_path = os.path.join(dataset_dir,
                                                train_params.tgt_word_embedding_filename)
    src_corpus_file_path = os.path.join(dataset_dir, train_params.src_corpus_filename)
    tgt_corpus_file_path = os.path.join(dataset_dir, train_params.tgt_corpus_filename)

    src_word2id, src_id2word, src_embed_matrix = ensure_vocab_embedding(
        src_tokenizer,
        src_vocab_file_path,
        src_word_embedding_file_path,
        src_corpus_file_path,
        encoder_params.embedding_dim,
        "Source")

    tgt_word2id, tgt_id2word, tgt_embed_matrix = ensure_vocab_embedding(
        tgt_tokenizer,
        tgt_vocab_file_path,
        tgt_word_embedding_file_path,
        tgt_corpus_file_path,
        decoder_params.embedding_dim,
        "Target")

    dataset = ParallelTextDataSet(src_tokenizer,
                                  tgt_tokenizer,
                                  src_corpus_file_path,
                                  tgt_corpus_file_path,
                                  encoder_params.max_seq_len,
                                  decoder_params.max_seq_len,
                                  src_word2id,
                                  tgt_word2id)
    data_loader = DataLoader(dataset,
                             batch_size=train_params.batch_size,
                             shuffle=True,
                             collate_fn=dataset.collate_func)

    encoder_params.vocab_size = len(src_word2id)
    encoder_params.device = device
    encoder = train_params.encoder(encoder_params)
    # Freeze word embedding weight
    encoder.init_embedding_weight(src_embed_matrix)

    decoder_params.vocab_size = len(tgt_word2id)
    decoder_params.device = device
    decoder = train_params.decoder(decoder_params)
    # Freeze word embedding weight
    decoder.init_embedding_weight(tgt_embed_matrix)

    model: nn.Module = Seq2Seq(encoder, decoder)
    model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.learning_rate)

    epoch = 0
    avg_loss = 0.
    for epoch in range(train_params.n_epochs):
        avg_loss = train_model(model, optimizer, loss_func, data_loader, device, train_params,
                               encoder_params, decoder_params, epoch + 1)

    save_dir_path = os.path.join(train_params.model_save_directory,
                                 get_checkpoint_dir_path(epoch + 1))
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    # save checkpoint for last epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, os.path.join(save_dir_path, 'checkpoint.tar'))


if __name__ == '__main__':
    print("***** Training start *****")
    main()
    print("***** Training end *****")
