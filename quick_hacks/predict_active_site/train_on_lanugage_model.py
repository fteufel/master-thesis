import sys
sys.path.append('../..')
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet

import pandas as pd
import torch
import torch.nn as nn
import argparse
import logging
import os
import time
from torch.utils.data import Dataset, DataLoader
from typing import Union, Dict, Sequence, List, Any
from pathlib import Path
import numpy as np
from tape import TAPETokenizer, ProteinBertModel, ProteinBertAbstractModel, visualization
from train_scripts.training_utils import SequenceTaggingDataset

from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S")
c_handler.setFormatter(formatter)
logger.addHandler(c_handler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define step as global
global_step = 0


class BertModelforBinaryTagging(ProteinBertAbstractModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ProteinBertModel(config = config)#, is_LM = False)
        setattr(config, 'dropout', 0.2)
        self.tagging_model = nn.Sequential(
                                    torch.nn.utils.weight_norm(nn.Linear(config.hidden_size, 256), dim=None), #hardcoded for now
                                    nn.ReLU(),
                                    nn.Dropout(config.dropout, inplace=True),
                                    torch.nn.utils.weight_norm(nn.Linear(256, 1), dim=None))

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets =None):
        outputs = self.encoder(input_ids, input_mask) #([batch_size, seq_len, dim], [batch_size,dim])
        sequence_output, _ = outputs
        #reshape to seq_len_first

        prediction_scores = self.tagging_model(sequence_output)
        outputs = (prediction_scores,)
        #loss, marginal_probs, best_path
        if targets is not None:
            # Binary crossentropy details: 
            # - needs float targets
            # - weight is tensor of same shape as targets. By  putting 0s at padded positions, I can ignore them
            loss = nn.functional.binary_cross_entropy_with_logits(
                                                                prediction_scores.reshape(-1), 
                                                                targets.reshape(-1).float(), 
                                                                weight=input_mask.reshape(-1).float(),
                                                                )
            #loss_fct = nn.BCEWithLogitsLoss(ignore_index=-1)
            #loss = loss_fct(
            #    prediction_scores.reshape(-1), targets.reshape(-1))

        outputs =  (loss,) + outputs

        return outputs



def train_model(model, dataloader, optimizer, args, visualizer):

    model.train()
    for i, batch in enumerate(dataloader):
        data, targets, mask = batch
        data, targets, mask = data.to(device), targets.to(device), mask.to(device)
        #TODO maybe permute
        loss, _ = model(data, mask, targets=targets)

        loss.backward()
        optimizer.step()

        train_metrics = {'loss': loss.item()}
        visualizer.log_metrics(train_metrics, "train", global_step)
        logger.info(train_metrics)
        global_step += 1



    return loss

def validate_model(model, dataloader, args, visualizer):

    model.eval()
    #va_probs_tot = np.zeros((0))
    #va_labels_tot = np.zeros((0))
    total_loss = 0
    total_auprc = 0
    for i, batch in enumerate(dataloader):
        data, targets, mask = batch
        data, targets, mask = data.to(device), targets.to(device), mask.to(device)
        #TODO maybe permute
        loss, scores = model(data, mask, targets=targets)

        total_loss += loss.item()

        x= targets.reshape(-1).cpu().detach().numpy()
        y =scores.reshape(-1).cpu().detach().numpy()
        total_auprc +=  average_precision_score(x[~x==-1],y[~x==-1])
        #print(f'{i} worked loss: {average_loss} {average_auprc}.')

       # va_probs_tot = np.concatenate([va_probs_tot, scores.reshape(-1).cpu().detach().numpy()])
       # va_labels_tot = np.concatenate([va_labels_tot, targets.reshape(-1).cpu().detach().numpy()])

    val_loss = total_loss/i
    va_auprc = total_auprc/i
    #va_prc = average_precision_score(va_labels_tot, va_probs_tot)


    val_metrics = {'loss': val_loss, 'AUPRC': va_prc}
    visualizer.log_metrics(val_metrics, "val", global_step)
    return val_loss


def main_training_loop(args):
    #setup model
    model = BertModelforBinaryTagging.from_pretrained('bert-base')

    data_train = SequenceTaggingDataset(args.train_data, label_dict= {'N':0, 'A':1})
    data_valid = SequenceTaggingDataset(args.valid_data, label_dict= {'N':0, 'A':1})

    dl_train = DataLoader(data_train, collate_fn=data_train.collate_fn, batch_size= args.batch_size)
    dl_valid = DataLoader(data_valid, collate_fn=data_valid.collate_fn, batch_size= args.batch_size)

    #training logger
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    experiment_name = f"{args.experiment_name}_{model.base_model_prefix}_{time_stamp}"
    viz = visualization.get(args.output_dir, experiment_name, local_rank = -1) #debug=args.debug) #this -1 means traning is not distributed, debug makes experiment dry run for wandb

    viz.log_config(args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_loss = 10000000000000
    for epoch in range(args.epochs):
        logger.info(f'Training epoch {epoch}')
        #loss = train_model(model, dl_train, optimizer, args, viz)

        va_loss = validate_model(model, dl_valid, args, viz)
        if va_loss < best_loss:
            best_loss = va_loss
            logger.info('Save model')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type = str)
    parser.add_argument('--valid_data', type = str)
    parser.add_argument('--experiment_name', type = str)
    parser.add_argument('--output_dir', type = str, default = 'training_run')

    parser.add_argument('--learning_rate', type = float, default = 0.001)
    parser.add_argument('--batch_size', type  = int, default= 5)
    parser.add_argument('--epochs', type  = int, default= 100)


    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main_training_loop(args)