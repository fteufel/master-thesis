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
from train_scripts.training_utils import FullSeqHdf5Dataset
import tape
from tape import TAPETokenizer, UniRepForLM
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
logger.addHandler(c_handler)

def load_and_eval_model(dataloader: torch.utils.data.DataLoader, model: tape.ProteinModel, checkpoint: str) -> float:
    '''Evaluate perplexity of Dataset with pretrained model.
    Assume that model forward() returns the loss
    Inputs:
        model: model that returns loss at [0], implements .from_pretrained static factory method
        dataset: dataset that gives individual sequences (not tbtt pretraining style subsequences)
        checkpoint: path/string to model checkpoint
    returns:
        Loss, averaged per sequence.
    '''
    #load model
    model = model.from_pretrained(checkpoint)

    model.eval()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        data, targets = batch
        data = data.to(device)
        targets = targets.to(device)

        loss, _, _ = model(data, targets = targets) #loss, output, hidden states
        total_loss += loss.item()
        print(f'{i}, {math.exp(loss.item())}')

    return total_loss / len(dataloader) #normalize by dataset size

parser = argparse.ArgumentParser(description='Evaluation of LMs.')
parser.add_argument('--data', type=str, default='data/awdlstmtestdata/',
                    help='location of the data corpus')
parser.add_argument('--checkpoint_dir', type=str, default='/zhome/1d/8/153438/experiments/results/',
                    help='location of the saved models, directories named as here in code')
parser.add_argument('--output_dir', type = str, default='perlexity_test',
                    help='output directory')
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


# setup models
#NOTE tape unirep is not the real unirep
checkpoint_dict = {
#'unirep': 'babbler-1900',
'euk_awdlstm' : os.path.join(args.checkpoint_dir,'best_euk_model'),
'pla_awdlstm' : os.path.join(args.checkpoint_dir,'best_pla_model')
}
model_dict = {
#'unirep': UniRepForLM,
#'euk_awdlstm' : ProteinAWDLSTMForLM,
'pla_awdlstm' : ProteinAWDLSTMForLM,
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Running on: {device}')

test_data = FullSeqHdf5Dataset(os.path.join(args.data,'plasmodium', 'valid.hdf5'))
testloader = DataLoader(test_data, 20, collate_fn=test_data.collate_fn)

plasm_perplexity_dict = {}
for model in model_dict:
    loss = load_and_eval_model(testloader, model_dict[model], checkpoint_dict[model])
    plasm_perplexity_dict[model] = np.exp(loss)
    logger.info(f'Model {model}, Plasmodium perplexity {math.exp(loss)}')

test_data = FullSeqHdf5Dataset(os.path.join(args.data,'eukarya', 'test.hdf5'))
testloader = DataLoader(test_data, 40, collate_fn=test_data.collate_fn)

euk_perplexity_dict = {}
for model in model_dict:
    loss = load_and_eval_model(testloader, model_dict[model], checkpoint_dict[model])
    euk_perplexity_dict[model] = np.exp(loss)
    logger.info(f'Model {model}, Eukarya perplexity {math.exp(loss)}')


df = pd.DataFrame.from_dict([checkpoint_dict,plasm_perplexity_dict, euk_perplexity_dict]).T.rename(columns ={0:'model', 1:'Perplexity Plasmodium',2:'Perplexity Eukarya'})
time_stamp = time.strftime("%y-%m-%d-%H_%M", time.gmtime())
df.to_csv(os.path.join(args.output_dir, f'test_set_perplexities{time_stamp}.csv'))