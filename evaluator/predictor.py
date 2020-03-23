"""Predictor"""
import torch
from utils.constants import (
    SOS_TOKEN, EOS_TOKEN, RNN_NAME
)

class Predictor(object):
    """Predictor class"""
    def __init__(self, model, src_vocab, src_vocab_query, trg_vocab, device):
        self.model = model
        self.src_vocab = src_vocab
        self.src_vocab_query = src_vocab_query
        self.trg_vocab = trg_vocab
        self.device = device

    def _predict_step(self, tokens, query_tokens):
        self.model.eval()
        tokenized_sentence = [SOS_TOKEN] + [t.lower() for t in tokens] + [EOS_TOKEN]
        numericalized = [self.src_vocab.stoi[token] for token in tokenized_sentence]
        src_tensor = torch.LongTensor(numericalized).unsqueeze(0).to(self.device)

        tokenized_query_sentence = [SOS_TOKEN] + [t.lower() for t in query_tokens] + [EOS_TOKEN]
        numericalized_query = [self.src_vocab_query.stoi[token] for token in tokenized_query_sentence]
        src_query_tensor = torch.LongTensor(numericalized_query).unsqueeze(0).to(self.device)

        with torch.no_grad():
            encoder_out = self.model.encoder(src_tensor)
            #encoder_out = torch.zeros([src_tensor.size()[0],src_tensor.size()[1],512]).to(device='cuda')
            encoder_out_query = self.model.encoder_query(src_query_tensor)
            #encoder_out_query = torch.zeros([src_query_tensor.size()[0],src_query_tensor.size()[1],512]).to(device='cuda')
            encoder_out = torch.cat((encoder_out, encoder_out_query), dim=1)

        outputs = [self.trg_vocab.stoi[SOS_TOKEN]]

        # cnn positional embedding gives assertion error for tensor
        # of size > max_positions-1, we predict tokens for max_positions-2
        # to avoid the error
        for _ in range(self.model.decoder.max_positions-2):
            trg_tensor = torch.LongTensor(outputs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model.decoder(trg_tensor, encoder_out, src_tokens=src_tensor, src_query_tokens=src_query_tensor)

            prediction = output.argmax(2)[:, -1].item()

            if prediction == self.trg_vocab.stoi[EOS_TOKEN]:
                break

            outputs.append(prediction)

        translation = [self.trg_vocab.itos[i] for i in outputs]

        return translation[1:] # , attention

    def _predict_rnn_step(self, tokens_ques, tokes_quer):
        self.model.eval()
        with torch.no_grad():
            tokenized_question = [SOS_TOKEN] + [t.lower() for t in tokens_ques] + [EOS_TOKEN]
            numericalized = [self.src_vocab.stoi[t] for t in tokenized_question]

            tokenized_query = [SOS_TOKEN] + [t.lower() for t in tokes_quer] + [EOS_TOKEN]
            numericalizedQuery = [self.src_vocab_query.stoi[t] for t in tokenized_query]

            src_len = torch.LongTensor([len(numericalized)]).to(self.device)
            src_len_query = torch.LongTensor([len(numericalizedQuery)]).to(self.device)
            tensor = torch.LongTensor(numericalized).unsqueeze(1).to(self.device)
            tensorQuery = torch.LongTensor(numericalizedQuery).unsqueeze(1).to(self.device)

            translation_tensor_logits = self.model(tensor.t(), src_len, tensorQuery.t(), src_len_query, None)

            translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
            translation = [self.trg_vocab.itos[t] for t in translation_tensor]

        return translation[1:] # , attention

    def predict(self, tokens_ques, tokes_quer):
        """Perform prediction on given tokens"""
        return self._predict_rnn_step(tokens_ques, tokes_quer) if self.model.name == RNN_NAME else \
                self._predict_step(tokens_ques, tokes_quer)
