"""
This example loads the pre-trained bert-base-nli-mean-tokens models from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.
"""
from torch.utils.data import DataLoader
import math
import torch
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, readers, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import VQUANDAReader
import logging
from datetime import datetime
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(os.environ['CUDA_VISIBLE_DEVICES'])

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

print(torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Read the dataset
model_name = 'bert-base-uncased'
train_batch_size = 30
num_epochs = 50
model_save_path = 'output/training_VQUANDA_continue_training-' + model_name + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

sts_reader = readers.VQUANDAReader.VQUANDAReader('/data/premnadh/sentence-transformers/examples/datasets')

# Use BERT for mapping tokens to embeddings
word_embedding_model = models.BERT(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

# Load a pre-trained sentence transformer model
#model = SentenceTransformer(model_name)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Convert the dataset to a DataLoader ready for training
logging.info("Read VQUANDA train dataset")
train_data = SentencesDataset(sts_reader.get_examples('/train.json'), model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)


logging.info("Read VQUANDA dev dataset")
dev_data = SentencesDataset(examples=sts_reader.get_examples('/test.json'), model=model)
k = len(dev_data.labels)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_data)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_data = SentencesDataset(examples=sts_reader.get_examples("/test.json"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
model.evaluate(evaluator)
