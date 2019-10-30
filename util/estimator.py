# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import nltk

from dataset import ParallelTextDataSet
from module import Seq2Seq
from util import AttributeDict
from util import get_checkpoint_dir_path
from util import index2word


class Estimator:
    class Mode(Enum):
        TRAIN = 0
        EVAL = 1
        INFERENCE = 2

    def __init__(self,
                 device: str,
                 common_params: AttributeDict):
        self.device = device
        self.common_params = common_params
        encoder_params = AttributeDict(self.common_params.encoder_params)
        decoder_params = AttributeDict(self.common_params.decoder_params)
        self.common_params.encoder_params = encoder_params
        self.common_params.decoder_params = decoder_params
        encoder_params.device = self.device
        decoder_params.device = self.device
        self.mode = None

        self.base_dir = os.getcwd()
        self.data_set_dir = os.path.join(self.base_dir, 'dataset')

        self.src_tokenizer = common_params.src_tokenizer()
        self.tgt_tokenizer = common_params.tgt_tokenizer()

        self.src_vocab_file_path = os.path.join(self.data_set_dir, common_params.src_vocab_filename)
        self.tgt_vocab_file_path = os.path.join(self.data_set_dir, common_params.tgt_vocab_filename)
        self.src_word_embedding_file_path = os.path.join(self.data_set_dir,
                                                         common_params.get(
                                                             'src_word_embedding_filename', None))
        self.tgt_word_embedding_file_path = os.path.join(self.data_set_dir,
                                                         common_params.get(
                                                             'tgt_word_embedding_filename', None))

        self.src_word2id, self.src_id2word, self.src_embedding_weight = self._build_vocab(
            self.src_vocab_file_path,
            self.src_word_embedding_file_path,
        )
        if encoder_params.get('vocab_size', None) is None:
            encoder_params.vocab_size = len(self.src_word2id)

        self.tgt_word2id, self.tgt_id2word, self.tgt_embedding_weight = self._build_vocab(
            self.tgt_vocab_file_path,
            self.tgt_word_embedding_file_path
        )
        if decoder_params.get('vocab_size', None) is None:
            decoder_params.vocab_size = len(self.tgt_word2id)

        self.model: nn.Module = self._build_model(self.common_params, self.device)

    def get_model_parameters(self):
        return self.model.parameters()

    @staticmethod
    def _build_model(params, device) -> nn.Module:
        model = None
        if params.model == Seq2Seq:
            encoder_params = params.encoder_params
            decoder_params = params.decoder_params
            encoder = encoder_params.model(encoder_params)
            decoder = decoder_params.model(decoder_params)
            model = params.model(encoder, decoder).to(device)
        else:
            raise AssertionError('Not implemented.. T^T')
        return model

    def train(self, train_params: AttributeDict, loss_func, optimizer):
        # Merge common and train params
        params = AttributeDict(self.common_params.copy())
        params.update(train_params)
        self._set_mode(Estimator.Mode.TRAIN)

        encoder_params = params.encoder_params
        decoder_params = params.decoder_params

        src_corpus_file_path = os.path.join(self.data_set_dir, params.src_corpus_filename)
        tgt_corpus_file_path = os.path.join(self.data_set_dir, params.tgt_corpus_filename)

        data_loader = self._prepare_data_loader(src_corpus_file_path, tgt_corpus_file_path, params,
                                                encoder_params.max_seq_len,
                                                decoder_params.max_seq_len)

        epoch = 0
        avg_loss = 0.
        for epoch in range(params.n_epochs):
            avg_loss = self._train_model(data_loader, params, self.model, loss_func, optimizer,
                                         self.device, epoch + 1)

        save_dir_path = os.path.join(train_params.model_save_directory,
                                     get_checkpoint_dir_path(epoch + 1))
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        # save checkpoint for last epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, os.path.join(save_dir_path, 'checkpoint.tar'))

    @staticmethod
    def _train_model(data_loader: DataLoader,
                     params: AttributeDict,
                     model: nn.Module,
                     loss_func,
                     optimizer,
                     device: str,
                     epoch: int):
        model.train()
        n_epochs = params.n_epochs
        losses = []
        data_length = len(data_loader)

        with tqdm(data_loader, total=data_length, desc=f'Epoch {epoch:03d}') as tqdm_iterator:
            for _, batch in enumerate(tqdm_iterator):
                src_seqs, src_lengths, tgt_seqs, tgt_lengths = batch
                src_seqs = src_seqs.to(device)
                src_lengths = src_lengths.to(device)
                tgt_seqs = tgt_seqs.to(device)
                tgt_lengths = tgt_lengths.to(device)

                _, loss = Estimator._train_step(model, src_seqs, src_lengths, tgt_seqs, tgt_lengths,
                                                loss_func, optimizer)
                losses.append(loss)
                tqdm_iterator.set_postfix_str(f'loss: {loss:05.3f}')

        avg_loss = np.mean(losses)
        print(f'Epochs [{epoch}/{n_epochs}] avg losses: {avg_loss:05.3f}', flush=True)
        return avg_loss

    @staticmethod
    def _forward_step(model: nn.Module,
                      src_seqs: torch.Tensor,
                      src_lengths: torch.Tensor,
                      tgt_seqs: torch.Tensor,
                      tgt_lengths: torch.Tensor):
        logits = model(src_seqs, src_lengths, tgt_seqs, tgt_lengths)
        return logits

    @staticmethod
    def _loss_step(model: nn.Module,
                   src_seqs: torch.Tensor,
                   src_lengths: torch.Tensor,
                   tgt_seqs: torch.Tensor,
                   tgt_lengths: torch.Tensor,
                   loss_func):

        logits = Estimator._forward_step(model, src_seqs, src_lengths, tgt_seqs, tgt_lengths)

        # To calculate loss, we should change shape of logits and labels
        # N is batch * seq_len, C is number of classes. (vocab size)
        # logits : (N by C)
        # labels : (N)
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = tgt_seqs.contiguous().view(-1)
        indices_except_padding = [labels != 0]
        loss = loss_func(logits[indices_except_padding], labels[indices_except_padding])
        return logits, loss

    @staticmethod
    def _train_step(model: nn.Module,
                    src_seqs: torch.Tensor,
                    src_lengths: torch.Tensor,
                    tgt_seqs: torch.Tensor,
                    tgt_lengths: torch.Tensor,
                    loss_func,
                    optimizer):
        logits, loss = Estimator._loss_step(model, src_seqs, src_lengths, tgt_seqs, tgt_lengths,
                                            loss_func)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return logits, loss.item()

    def _prepare_data_loader(self, src_corpus_path, tgt_corpus_path, params, src_max_seq_len,
                             tgt_max_seq_len):
        data_set = ParallelTextDataSet(self.src_tokenizer, self.tgt_tokenizer, src_corpus_path,
                                       tgt_corpus_path, src_max_seq_len, tgt_max_seq_len,
                                       self.src_word2id, self.tgt_word2id)
        data_loader = DataLoader(data_set, batch_size=params.batch_size, shuffle=True,
                                 collate_fn=data_set.collate_func)
        return data_loader

    def eval(self, eval_params: AttributeDict, loss_func):
        self._set_mode(Estimator.Mode.EVAL)
        params = self.common_params.copy()
        params.update(eval_params)
        encoder_params = params.encoder_params
        decoder_params = params.decoder_params

        src_corpus_file_path = os.path.join(self.data_set_dir, params.src_corpus_filename)
        tgt_corpus_file_path = os.path.join(self.data_set_dir, params.tgt_corpus_filename)

        data_loader = self._prepare_data_loader(src_corpus_file_path, tgt_corpus_file_path, params,
                                                encoder_params.max_seq_len,
                                                decoder_params.max_seq_len)
        avg_loss, bleu_score = self._eval_model(data_loader, params, self.model, loss_func,
                                                self.device, self.tgt_id2word)
        print(f'Avg loss: {avg_loss:05.3f}, BLEU score: {bleu_score}')

    @staticmethod
    def _eval_model(data_loader: DataLoader,
                    params: AttributeDict,
                    model: nn.Module,
                    loss_func,
                    device: str,
                    tgt_id2word: dict):
        with torch.no_grad():
            losses = []
            data_length = len(data_loader)

            predicted_sentences = []
            target_sequences = []

            with tqdm(data_loader, total=data_length, desc='EVAL') as tqdm_iterator:
                for _, batch in enumerate(tqdm_iterator):
                    src_seqs, src_lengths, tgt_seqs, tgt_lengths = batch
                    src_seqs = src_seqs.to(device)
                    src_lengths = src_lengths.to(device)
                    tgt_seqs = tgt_seqs.to(device)
                    tgt_lengths = tgt_lengths.to(device)

                    logits, loss = Estimator._loss_step(model, src_seqs, src_lengths, tgt_seqs,
                                                        tgt_lengths, loss_func)
                    predictions = Estimator.change_prediction_tensor(logits)

                    for predicted_sentence in predictions:
                        predicted_sentences.append(index2word(tgt_id2word, predicted_sentence))

                    for tgt_seq in tgt_seqs:
                        target_sequences.append(index2word(tgt_id2word, tgt_seq))

                    losses.append(loss)
                    tqdm_iterator.set_postfix_str(f'loss: {loss:05.3f}')
            bleu_score = nltk.translate.bleu_score.corpus_bleu(target_sequences,
                                                               predicted_sentences)
            return np.mean(losses), bleu_score

    @staticmethod
    def change_prediction_tensor(self, logits: torch.Tensor):
        _, indices = torch.max(logits.detach().cpu(), dim=-1, keepdim=True)
        predictions = indices.squeeze(-1)
        return predictions

    def inference(self, inputs):
        self._set_mode(Estimator.Mode.INFERENCE)
        # TODO: 문장 또는 문장 전체가 들어올 경우도 고려.
        pass

    def _set_mode(self, mode: Mode):
        self.mode = mode
        if mode == Estimator.Mode.TRAIN:
            self.model.train()
        else:
            self.model.eval()

    @staticmethod
    def _build_vocab(vocab_file_path: str, word_embedding_file_path: str):
        """
        :return: (word2id, id2word)
        """
        print(f'build_vocab vocab_file_path: {vocab_file_path}')

        with open(vocab_file_path, mode='r', encoding='utf-8') as f:
            tokens = f.readlines()
        original_tokens_count = len(tokens)
        print(f'  tokens count: {original_tokens_count}')

        embedding_matrix = None
        if word_embedding_file_path is not None:
            embedding_matrix = np.load(word_embedding_file_path)
            assert embedding_matrix.shape[0] == original_tokens_count, \
                'vocab count and embedding weight is not same. you MUST re-create embedding weight'

        word2id = {}
        id2word = {}
        invalid_indices = []
        for index, token in enumerate(tokens):
            stripped_token = token.lstrip().rstrip()
            if len(stripped_token) == 0:
                print(f'  This token is invalid. index: {index}, token: {token}')
                invalid_indices.append(index)
                continue
            word2id[stripped_token] = index
            id2word[index] = stripped_token

        assert len(invalid_indices) == 0, \
            f'You have untrusted vocabulary file. check indices {invalid_indices}'

        # if len(invalid_indices) > 0:
        # vocab_words = [f'{token}\n' for token in word2id.keys()]
        # with open(vocab_file_path, mode='w', encoding='utf-8') as wf:
        #     wf.writelines(vocab_words)
        # word2id.clear()
        # id2word.clear()
        # print(f'Now, this vocab has no invalid token')
        # with open(vocab_file_path, mode='r', encoding='utf-8') as f:
        #     tokens = f.readlines()
        # for index, token in enumerate(tokens):
        #     stripped_token = token.lstrip().rstrip()
        #     if len(stripped_token) == 0:
        #         continue
        #     word2id[token] = index
        #     id2word[index] = token

        # print(f'  dic count: {len(word2id)}')
        # if len(invalid_indices) > 0 and embedding_matrix is not None:
        #     embedding_matrix = np.delete(embedding_matrix, invalid_indices, axis=0)
        #     assert len(word2id) == embedding_matrix.shape[0], \
        #         'This should not be happened. check estimator logic.'
        #     os.remove(word_embedding_file_path)
        #     np.save(word_embedding_file_path, embedding_matrix)

        return word2id, id2word, embedding_matrix
