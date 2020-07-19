"""Run baseline model"""
import os
import math
import random
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator

from models.seq2seq import Seq2Seq
from utils.checkpoint import Chechpoint
from trainer.trainer import Trainer
from utils.scorer import BleuScorer
from evaluator.predictor import Predictor
from utils.verbaldataset import VerbalDataset
from utils.constants import (
    SEED, CUDA, CPU, PAD_TOKEN, RNN_NAME, CNN_NAME,
    TRANSFORMER_NAME, ATTENTION_1, ATTENTION_2, QUESTION, QUERY
)

DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)

ques = 0
query = 1

def parse_args():
    """Add arguments to parser"""
    parser = argparse.ArgumentParser(description='Verbalization dataset baseline models.')
    parser.add_argument('--model', default=TRANSFORMER_NAME, type=str,
                        choices=[RNN_NAME, CNN_NAME, TRANSFORMER_NAME], help='model to train the dataset')
    parser.add_argument('--input', default=QUERY, type=str,
                        choices=[QUESTION, QUERY], help='use query as input')
    parser.add_argument('--attention', default=ATTENTION_1, type=str,
                        choices=[ATTENTION_1, ATTENTION_2], help='attention layer for rnn model')
    parser.add_argument('--cover_entities', action='store_true', help='cover entities')
    parser.add_argument('--batch_size', default=30, type=int, help='batch size')
    parser.add_argument('--epochs_num', default=50, type=int, help='number of epochs')
    args = parser.parse_args()
    return args

def save_vocab(vocab):
    path = '/data/premnadh/VQUANDA-Baseline-Models-1/trg_vocab.txt'
    with open(path, 'w+') as f:     
        for token, index in vocab.stoi.items():
            f.write(f'{token}\n')

def main():
    """Main method to run the models"""
    args = parse_args()
    dataset = []
    vocab = []
    whole_data = []
    for x in [ques,query]:
        dataset.append(VerbalDataset())
        #changed - cover_entities
        dataset[x].load_data_and_fields(cover_entities=True, query_as_input=x)
        vocab.append(dataset[x].get_vocabs())
        whole_data.append(dataset[x].get_data())

    src_vocab, trg_vocab = vocab[0]
    src_vocab_query, trg_vocab_query = vocab[1]
    train_data_question, valid_data_question, test_data_question = whole_data[0]
    print("train_data_quer", len(list(train_data_question)))
    train_data_query, valid_data_query, test_data_query = whole_data[1]

    save_vocab(trg_vocab)

    print('--------------------------------')
    print(f'Model: {args.model}')
    print(f'Model input: {args.input}')
    if args.model == RNN_NAME:
        print(f'Attention: {args.attention}')
    print(f'Cover entities: {args.cover_entities}')
    print('--------------------------------')
    print(f"Training data: {len(train_data_query.examples)}")
    print(f"Evaluation data: {len(valid_data_query.examples)}")
    print(f"Testing data: {len(test_data_query.examples)}")
    print('--------------------------------')
    print(f'Question example: {train_data_query.examples[0].src}')
    print(f'Answer example: {train_data_query.examples[0].trg}')
    print('--------------------------------')
    print(f"Unique tokens in questions vocabulary: {len(src_vocab_query)}")
    print(f"Unique tokens in answers vocabulary: {len(trg_vocab_query)}")
    print('--------------------------------')
    print(f'Batch: {args.batch_size}')
    print(f'Epochs: {args.epochs_num}')
    print('--------------------------------')

    if args.model == RNN_NAME and args.attention == ATTENTION_1:
        from models.rnn1 import Encoder, Decoder
    elif args.model == RNN_NAME and args.attention == ATTENTION_2:
        from models.rnn2 import Encoder, Decoder
    elif args.model == CNN_NAME:
        from models.cnn import Encoder, Decoder
    elif args.model == TRANSFORMER_NAME:
        from models.transformer import Encoder, Decoder, NoamOpt

    # create model
    encoder = Encoder(src_vocab, DEVICE)
    encoder_query = Encoder(src_vocab_query, DEVICE)
    decoder = Decoder(trg_vocab_query, DEVICE)
    model = Seq2Seq(encoder, encoder_query, decoder, args.model).to(DEVICE)

    parameters_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters_num:,} trainable parameters')
    print('--------------------------------')

    # create optimizer
    if model.name == TRANSFORMER_NAME:
        # initialize model parameters with Glorot / fan_avg
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        optimizer = NoamOpt(torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        optimizer = optim.Adam(model.parameters())

    # define criterion
    criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.stoi[PAD_TOKEN])

    # train data
    trainer = Trainer(optimizer, criterion, args.batch_size, DEVICE)
    trainer.train(model, train_data_question, train_data_query, valid_data_question, valid_data_query, num_of_epochs=args.epochs_num)

    # load model
    model = Chechpoint.load(model)

    # generate test iterator
    valid_iterator, test_iterator = BucketIterator.splits(
                                        (valid_data_question, test_data_question),
                                        repeat=False,
                                        batch_size=args.batch_size,
                                        sort_within_batch=True if args.model == RNN_NAME else False,
                                        sort_key=lambda x: len(x.src),
                                        device=DEVICE)

    valid_iterator_query, test_iterator_query = BucketIterator.splits(
                                        (valid_data_query, test_data_query),
                                        repeat=False,
                                        batch_size=args.batch_size,
                                        sort_within_batch=True if args.model == RNN_NAME else False,
                                        sort_key=lambda x: len(x.src),
                                        device=DEVICE)
    
    
    # evaluate model
    valid_loss = trainer.evaluator.evaluate(model, valid_iterator, valid_iterator_query)
    test_loss = trainer.evaluator.evaluate(model, test_iterator, test_iterator_query)


    # calculate blue score for valid and test data
    predictor = Predictor(model, src_vocab, src_vocab_query, trg_vocab, DEVICE)
    valid_scorer = BleuScorer()
    test_scorer = BleuScorer()
    valid_scorer.data_score(valid_data_question.examples, valid_data_query.examples, predictor)
    results, _ = test_scorer.data_score(test_data_question.examples, test_data_query.examples, predictor)

    for k in results[0:10]:
        print("reference ", k['reference'])
        print("hypothesis", k['hypothesis'])

    print(f'| Val. Loss: {valid_loss:.3f} | Test PPL: {math.exp(valid_loss):7.3f} |')
    print(f'| Val. Data Average BLEU score {valid_scorer.average_score()} |')
    print(f'| Val. Data Average METEOR score {valid_scorer.average_meteor_score()} |')
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    print(f'| Test Data Average BLEU score {test_scorer.average_score()} |')
    print(f'| Test Data Average METEOR score {test_scorer.average_meteor_score()} |')


if __name__ == "__main__":
    # set a seed value
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    main()
