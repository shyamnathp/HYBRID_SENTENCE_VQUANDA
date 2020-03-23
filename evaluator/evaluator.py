import torch
from utils.constants import RNN_NAME

class Evaluator(object):
    """Evaluator class"""
    def __init__(self, criterion):
        self.criterion = criterion

    def evaluate(self, model, iterator, iterator_query, teacher_ratio=1.0):
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for _, batch in enumerate(zip(iterator, iterator_query)):
                batch_ques, batch_query = batch
                src_ques, src_len_ques = batch_ques.src
                src_query, src_len_query = batch_query.src
                trg = batch_query.trg
                input_trg = trg if model.name == RNN_NAME else trg[:, :-1]
                output = model(src_ques, src_len_ques, src_query, src_len_query, input_trg, teacher_ratio)
                trg = trg.t() if model.name == RNN_NAME else trg[:, 1:]
                output = output.contiguous().view(-1, output.shape[-1])
                trg = trg.contiguous().view(-1)
                # output: (batch_size * trg_len) x output_dim
                # trg: (batch_size * trg_len)
                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
        return epoch_loss / len(iterator)