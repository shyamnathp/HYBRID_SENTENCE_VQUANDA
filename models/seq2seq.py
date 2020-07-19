"""
Main sequence to sequence class which conects
encoder-decoder model
"""
import torch
import torch.nn as nn
from transformers import BertModel
from sentence_transformers import SentenceTransformer

class Seq2Seq(nn.Module):
    """
    Seq2seq class
    """
    def __init__(self, encoder, encoder_query, decoder, name):
        super().__init__()
        #self.sentencetransformer = SentenceTransformer('bert-base-nli-mean-tokens')
        #self.bert = BertModel.from_pretrained('/data/premnadh/.cache/torch/sentence_transformers/public.ukp.informatik.tu-darmstadt.de_reimers_sentence-transformers_v0.2_bert-base-nli-mean-tokens.zip/0_BERT')
        #self.encoder_query = encoder_query
        self.encoder = encoder
        self.decoder = decoder
        self.name = name

    def forward(self, src_tokens, src_lengths, src_query_tokens, src_query_lengths, trg_tokens, teacher_forcing_ratio=0.5):
        """
        Run the forward pass for an encoder-decoder model.

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(src_len, batch)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            trg_tokens (LongTensor): tokens in the target language of shape
                `(tgt_len, batch)`, for teacher forcing
            teacher_forcing_ratio (float): teacher forcing probability

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - attention scores of shape `(batch, trg_len, src_len)`
        """
        #encoder_out_bert = self.sentencetransformer.encode(src_tokens, batch_size=20)
        #output_tokens = self.bert(input_ids=features['input_ids'], token_type_ids=features['token_type_ids'], attention_mask=features['input_mask'])[0]
        #features = self.encode(src_tokens, src_lengths)
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        encoder_out_query = self.encoder(src_query_tokens, src_lengths=src_query_lengths)
        encoder_out = torch.cat((encoder_out, encoder_out_query), dim=1)
        decoder_out = self.decoder(trg_tokens, encoder_out,
                                   src_tokens=src_tokens, src_query_tokens=src_query_tokens,
                                   teacher_forcing_ratio=teacher_forcing_ratio)
        return decoder_out

    def encode(self, src_tokens, src_lengths, batch_size=100):
        src_tokens = src_tokens.tolist()
        max_length = torch.max(src_lengths).item()
        features = {}
        for src_token in src_tokens:
            src_token_check = src_token
        return features 
