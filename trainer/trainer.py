"""Trainer"""
import time
import torch
import numpy as np
from torchtext.data import BucketIterator
from evaluator.evaluator import Evaluator
from utils.checkpoint import Chechpoint
from utils.constants import RNN_NAME

class Trainer(object):
    """Trainer Class"""
    def __init__(self, optimizer, criterion, batch_size, device):
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = device
        self.evaluator = Evaluator(criterion=self.criterion)

    def _train_batch(self, model, iterator, iteratorQuery, teacher_ratio, clip):
        model.train()
        epoch_loss = 0
        for _, batch in enumerate(zip(iterator, iteratorQuery)):
            batch_ques, batch_query = batch
            src_ques, src_len_ques = batch_ques.src
            src_query, src_len_query = batch_query.src
            trg = batch_query.trg
            self.optimizer.zero_grad()
            input_trg = trg if model.name == RNN_NAME else trg[:, :-1]
            output = model(src_ques, src_len_ques, src_query, src_len_query, input_trg, teacher_ratio)
            trg = trg.t() if model.name == RNN_NAME else trg[:, 1:]
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg.contiguous().view(-1)
            # output: (batch_size * trg_len) x output_dim
            # trg: (batch_size * trg_len)
            loss = self.criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    def _get_iterators(self, train_data, valid_data, model_name):
        return BucketIterator.splits((train_data, valid_data),
                                     repeat=False,
                                     batch_size=self.batch_size,
                                     sort_within_batch=False,
                                     sort_key=lambda x: len(x.src),
                                     device=self.device)

    def _epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def _log_epoch(self, train_loss, valid_loss, epoch, start_time, end_time):
        minutes, seconds = self._epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Time: {minutes}m {seconds}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {np.exp(valid_loss):7.3f}')

    def _train_epoches(self, model, train_data, train_data_query, valid_data, valid_data_query, num_of_epochs, teacher_ratio, clip):
        best_valid_loss = float('inf')
        # pylint: disable=unbalanced-tuple-unpacking
        train_iterator, valid_iterator = self._get_iterators(train_data, valid_data, model.name)
        train_iterator_query, valid_iterator_query = self._get_iterators(train_data_query, valid_data_query, model.name)
        for epoch in range(num_of_epochs):
            start_time = time.time()
            train_loss = self._train_batch(model, train_iterator, train_iterator_query, teacher_ratio, clip)
            valid_loss = self.evaluator.evaluate(model, valid_iterator, valid_iterator_query, teacher_ratio)
            end_time = time.time()
            self._log_epoch(train_loss, valid_loss, epoch, start_time, end_time)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                Chechpoint.save(model)

    def train(self, model, train_data, train_data_query, valid_data, valid_data_query, num_of_epochs=20, teacher_ratio=1.0, clip=1):
        """Train model"""
        self._train_epoches(model, train_data, train_data_query, valid_data, valid_data_query, num_of_epochs, teacher_ratio, clip)