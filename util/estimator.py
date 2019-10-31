# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from enum import Enum

import nltk
import numpy as np
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ParallelTextDataSet
from module import Seq2Seq
from util import AttributeDict
from util import get_checkpoint_filename
from util import index2word
from util import pad_token
from util.tokens import EOS_TOKEN
from util.tokens import PAD_TOKEN
from util.tokens import UNK_TOKEN


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

        src_word_embedding_filename = common_params.get('src_word_embedding_filename', None)
        if src_word_embedding_filename is not None:
            self.src_word_embedding_file_path = os.path.join(self.data_set_dir,
                                                             src_word_embedding_filename)
        else:
            self.src_word_embedding_file_path = None

        tgt_word_embedding_filename = common_params.get('tgt_word_embedding_filename', None)
        if tgt_word_embedding_filename is not None:
            self.tgt_word_embedding_file_path = os.path.join(self.data_set_dir,
                                                             tgt_word_embedding_filename)
        else:
            self.tgt_word_embedding_file_path = None

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
        if self.src_word_embedding_file_path is not None:
            self.model.encoder.init_embedding_weight(np.load(self.src_word_embedding_file_path))
        if self.tgt_word_embedding_file_path is not None:
            self.model.decoder.init_embedding_weight(np.load(self.tgt_word_embedding_file_path))
        self.model.to(device)

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
            model = params.model(encoder, decoder)
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
                                     get_checkpoint_filename(epoch + 1))
        if os.path.exists(save_dir_path):
            os.remove(save_dir_path)

        # save checkpoint for last epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, save_dir_path)

    def _train_model(self,
                     data_loader: DataLoader,
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

                _, loss = self._train_step(model, src_seqs, src_lengths, tgt_seqs, tgt_lengths,
                                           loss_func, optimizer)
                losses.append(loss)
                tqdm_iterator.set_postfix_str(f'loss: {loss:05.3f}')

        avg_loss = np.mean(losses)
        print(f'Epochs [{epoch}/{n_epochs}] avg losses: {avg_loss:05.3f}', flush=True)
        return avg_loss

    def _forward_step(self,
                      model: nn.Module,
                      src_seqs: torch.Tensor,
                      src_lengths: torch.Tensor,
                      tgt_seqs: torch.Tensor,
                      tgt_lengths: torch.Tensor):
        logits = model(src_seqs, src_lengths, tgt_seqs, tgt_lengths)
        return logits

    def _loss_step(self,
                   model: nn.Module,
                   src_seqs: torch.Tensor,
                   src_lengths: torch.Tensor,
                   tgt_seqs: torch.Tensor,
                   tgt_lengths: torch.Tensor,
                   loss_func):
        tgt_seqs_input = tgt_seqs[:, :-1]
        tgt_seqs_ = tgt_seqs[:, 1:]
        logits = self._forward_step(model, src_seqs, src_lengths, tgt_seqs_input, tgt_lengths)

        # To calculate loss, we should change shape of logits and labels
        # N is batch * seq_len, C is number of classes. (vocab size)
        # logits : (N by C)
        # labels : (N)
        logits_flattened = logits.contiguous().view(-1, logits.size(-1))
        labels = tgt_seqs_.contiguous().view(-1)
        loss = loss_func(logits_flattened, labels)
        # indices_except_padding = [labels != self.src_word2id[PAD_TOKEN]]
        # loss = loss_func(logits_flattened[indices_except_padding], labels[indices_except_padding])
        return logits, loss

    def _train_step(self,
                    model: nn.Module,
                    src_seqs: torch.Tensor,
                    src_lengths: torch.Tensor,
                    tgt_seqs: torch.Tensor,
                    tgt_lengths: torch.Tensor,
                    loss_func,
                    optimizer):
        logits, loss = self._loss_step(model, src_seqs, src_lengths, tgt_seqs, tgt_lengths,
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

    def _load_checkpoint(self, params: AttributeDict):
        return torch.load(os.path.join(self.base_dir, params.checkpoint_path))

    def _load_checkpoint_from_path(self, checkpoint_path: str):
        return torch.load(os.path.join(self.base_dir, checkpoint_path))

    def eval(self, eval_params: AttributeDict, loss_func):
        self._set_mode(Estimator.Mode.EVAL)
        params = AttributeDict(self.common_params.copy())
        params.update(eval_params)
        encoder_params = params.encoder_params
        decoder_params = params.decoder_params

        # load checkpoint
        checkpoint = self._load_checkpoint(params)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        src_corpus_file_path = os.path.join(self.data_set_dir, params.src_corpus_filename)
        tgt_corpus_file_path = os.path.join(self.data_set_dir, params.tgt_corpus_filename)

        data_loader = self._prepare_data_loader(src_corpus_file_path, tgt_corpus_file_path, params,
                                                encoder_params.max_seq_len,
                                                decoder_params.max_seq_len)
        avg_loss, bleu_score = self._eval_model(data_loader, params, self.model, loss_func,
                                                self.device, self.tgt_id2word)
        print(f'Avg loss: {avg_loss:05.3f}, BLEU score: {bleu_score}')

    def _eval_model(self,
                    data_loader: DataLoader,
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

                    logits, loss = self._loss_step(model, src_seqs, src_lengths, tgt_seqs,
                                                   tgt_lengths, loss_func)
                    predictions = Estimator.change_prediction_tensor(logits)

                    for predicted_sentence in predictions:
                        predicted_sentences.append(index2word(tgt_id2word, predicted_sentence))

                    for tgt_seq in tgt_seqs:
                        target_sequences.append(index2word(tgt_id2word, tgt_seq))

                    losses.append(loss.item())
                    tqdm_iterator.set_postfix_str(f'loss: {loss:05.3f}')

            smoother = SmoothingFunction()
            bleu_score = nltk.translate.bleu_score.corpus_bleu(target_sequences,
                                                               predicted_sentences,
                                                               smoothing_function=smoother.method1)
            return np.mean(losses), bleu_score

    @staticmethod
    def change_prediction_tensor(logits: torch.Tensor):
        _, indices = torch.max(logits.detach().cpu(), dim=-1, keepdim=True)
        predictions = indices.squeeze(-1)
        return predictions

    def inference(self, inputs: str, eval_params: AttributeDict):
        self._set_mode(Estimator.Mode.INFERENCE)
        # self.device = 'cpu'
        # self.model.to(self.device)

        # load checkpoint
        checkpoint = self._load_checkpoint(eval_params)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        src_max_len = self.common_params.encoder_params.max_seq_len
        src_tokens = self.src_tokenizer.tokenize(inputs)
        src_lengths = len(src_tokens)

        for i, token in enumerate(src_tokens):
            if i == src_max_len - 1:
                break
            if token in self.src_word2id:
                src_tokens[i] = self.src_word2id[token]
            else:
                src_tokens[i] = self.src_word2id[UNK_TOKEN]
        src_tokens.append(self.src_word2id[EOS_TOKEN])

        pad_token(src_tokens, src_max_len)
        print(f'src padded tokens: {src_tokens}')

        with torch.no_grad():
            src_padded_tokens = torch.tensor(src_tokens, device=self.device)
            src_padded_tokens = src_padded_tokens.unsqueeze(0)
            src_lengths = torch.tensor(src_lengths).unsqueeze(0)

            logits = self._forward_step(self.model, src_padded_tokens, src_lengths, None, None)
            _, indices = torch.max(logits, dim=-1, keepdim=True)
            predictions = indices.squeeze(-1).squeeze(0)

            print(f'predicted tokens: {predictions.tolist()}')
            sentence = index2word(self.tgt_id2word, predictions)
            print(f'predicted sentence: {sentence}')

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

        return word2id, id2word, embedding_matrix
