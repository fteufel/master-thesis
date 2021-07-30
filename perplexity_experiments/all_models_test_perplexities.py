'''
Evaluate multiple different AWD-LSTM checkpoints in a directory on different datasets.
Names of checkpoints need to be in MODEL_LIST
Names of datasets need to be in TAXON_LIST
'''
import argparse
import time
import math
import numpy as np
import torch
from typing import Union
import logging
import torch.nn as nn
import sys
sys.path.append("..")
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet
from models.awd_lstm import ProteinAWDLSTMForLM, ProteinAWDLSTMConfig
from train_scripts.training_utils import FullSeqHdf5Dataset, VirtualBatchTruncatedBPTTHdf5Dataset, repackage_hidden
import tape
from tape import TAPETokenizer, UniRepForLM, ProteinModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S")
c_handler.setFormatter(formatter)
logger.addHandler(c_handler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Running on: {device}')

#eukarya  finetuned_phytophthora  finetuned_plasmodium  finetuned_wasps  plasmodium  wasps
#~/experiments/results/best_models_25_08_2020

TAXON_LIST  = ['Plasmodium', 'custom_taxon_wasp', 'Phytophthora'] #note we do not evaluate on eukarya, the test set is too big to process in the same run.
MODEL_LIST = ['plasmodium', 
              'wasps', 
              'phytophthora', 
              'eukarya', 
              'finetuned_plasmodium',
              'finetuned_wasps',
              'finetuned_phytophthora'
              ]

def load_and_eval_model(dataset: Dataset, model: ProteinModel, checkpoint: str) -> float:
    '''Evaluate perplexity of Dataset with pretrained model.
    Assume that model forward() returns the loss
    Inputs:
        model: model that returns loss at [0], implements .from_pretrained() static factory method
        dataset: dataset that gives individual sequences (not tbtt pretraining style subsequences)
        checkpoint: path/string to model checkpoint
    returns:
        Loss, averaged per sequence.
    '''
    #load data
    batch_size = 50
    dataloader = DataLoader(dataset, batch_size, collate_fn=dataset.collate_fn)

    #load model
    model = model.from_pretrained(checkpoint)
    model.to(device)

    model.eval()
    total_loss = 0
    for i, batch in enumerate(dataloader):

        data, targets = batch
        data = data.to(device).contiguous()
        targets = targets.to(device).contiguous()

        if data.dtype is not torch.int64:
            data, targets = data.to(torch.int64), targets.to(torch.int64)

        with torch.no_grad():
            regularized_loss, loss, _, _ = model(data, targets = targets) #loss, output, hidden states
            
        total_loss += loss.item()*data.shape[1] # * number of sequences in batch. Just because last batch might not be full.

    return total_loss / len(dataset) #normalize by dataset size


parser = argparse.ArgumentParser(description='Evaluation of LMs.')
parser.add_argument('--data', type=str, default='/work3/felteu/data/all_organisms_balanced_26082020/',
                    help='base directory of the datasets')
parser.add_argument('--checkpoint_dir', type=str, default='/work3/felteu/clean_data_models/',
                    help='location of the saved models, directories named as here in code')
parser.add_argument('--output_dir', type = str, default='perplexity_test',
                    help='output directory')
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

model = ProteinAWDLSTMForLM

results_dict = {}
for taxon in TAXON_LIST:
    data = FullSeqHdf5Dataset(os.path.join(args.data, taxon, 'test.hdf5'))

    perpl_list = []
    for checkpoint in MODEL_LIST:
        model_path = os.path.join(args.checkpoint_dir, checkpoint)
        lm_loss = load_and_eval_model(data,model,model_path)
        logger.info(f'Model {checkpoint}, {taxon} perplexity {math.exp(lm_loss)}')

        perpl_list.append(math.exp(lm_loss))

    results_dict[taxon] = perpl_list

df = pd.DataFrame.from_dict(results_dict, orient ='index', columns = MODEL_LIST)

time_stamp = time.strftime("%y-%m-%d-%H_%M", time.gmtime())
df.to_csv(os.path.join(args.output_dir, f'test_set_perplexities{time_stamp}.csv'))