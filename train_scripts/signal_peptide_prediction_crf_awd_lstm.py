# Script to fine-tune a pretrained AWD-LSTM model.
#like normal training loop, but optionally uses slanted triangular learning rate schedule
#Felix July 2020
import argparse
import time
import math
import numpy as np
import torch
import logging
import torch.nn as nn
import sys
sys.path.append("..")
from typing import Tuple
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet
from models.awd_lstm import ProteinAWDLSTMConfig
from models.sp_tagging_awd_lstm import ProteinAWDLSTMCRF
from tape import visualization #import utils.visualization as visualization
from utils.signalp_dataset import ThreeLineFastaDataset
from torch.utils.data import DataLoader
from apex import amp

import data
import os 
import random
import hashlib

from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S")
c_handler.setFormatter(formatter)
logger.addHandler(c_handler)


def training_step(model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, optimizer: torch.optim.Optimizer, 
                    args: argparse.ArgumentParser, i: int) -> (float, tuple):
    '''Predict one minibatch and performs update step.
    Returns:
        loss: loss value of the minibatch
    '''
    data = data.to(device)
    targets = targets.to(device)
    global_targets = (targets == 0).any(axis =1) *1 #binary signal peptide existence indicator


    with torch.autograd.detect_anomaly():
        model.train()
        optimizer.zero_grad()


        loss, _, _ = model(data, targets= targets, global_targets = global_targets)

        if torch.cuda.is_available():
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    return loss.item()

def compute_sp_detection(outputs, targets):
    '''Calculate binary detection (SP yes/no) from sequence output
    To make a single global prediction of whether a signal peptide is present or not in a protein, 
    we take the average of the marginal probabilities across the sequence 
    (nine classes: Sec/SPI signal, Tat/SPI signal, Sec/SPII signal, outer region, inner region, TM in-out, TM out-in, 
    Sec SPI/Tat SPI cleavage site and Sec/SPII cleavage site) and perform an affine linear transformation into 
    four classes (Sec/SPI, Sec/SPII, Tat/SPI, Other),ls=Ws[1T∑Tt=1p(yt|x)], 
    so as to get the logit of a categorical distribution over the presence or not of a signal peptide.
    '''
    outputs = outputs.detach().cpu().numpy() #(batch_size, seq_len, num_labels)
    targets = targets.detach().cpu().numpy() #(batch_size, seq_len)
    # SP (Sec/SPI) is token id 0
    targets_1d = (targets == 0).any(axis =1) #targets to binary per-sequence target label
    predictions = outputs.argmax(axis =2) #don't define threshold yet, the one with max score is the label for the position.

    #only extract positions with SP tag
    #outputs = 
    #targets = 


def validate(model: torch.nn.Module, valid_data: DataLoader) -> float:
    '''Run over the validation data. Average loss over the full set.
    '''
    model.eval()

    all_targets = []
    all_global_preds = []
    all_pos_preds = []

    total_loss = 0
    for i, batch in enumerate(valid_data):
        data, targets = batch
        data = data.to(device)
        targets = targets.to(device)
        global_targets = (targets == 0).any(axis =1) *1 #binary signal peptide existence indicator

        with torch.autograd.detect_anomaly(): #this seems to fix NaNs - does not make any sense logically, but ok.
            loss, global_pred, pos_preds = model(data, targets= targets, global_targets = global_targets)

        total_loss += loss.item()

        all_targets.append(targets.detach().cpu().numpy())
        all_global_preds.append(global_pred.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())

    #global_pred (batch_size, 4) ... 0 is SP indicator
    sp_score = global_pred[:,0]
    sp_true_presence = (targets == 0).any(axis =1) *1

    all_targets = np.concatenate(all_targets)
    all_global_preds = np.concatenate(all_global_preds)

    sp_score = all_global_preds[:,0]
    sp_true_presence = (all_targets ==0).any(axis =1) *1

    y_pred_thresholded = (sp_score >= 0.5) *1 

    #from IPython import embed
    #embed()
    logger.info(f'Globals {all_global_preds.sum()}')
    logger.info(f'Targets {all_targets.sum()}')
    logger.info(f'sp_true {sp_true_presence.sum()}')
    logger.info(f'sp true score {sp_score.sum()}')

    prc = average_precision_score(sp_true_presence, sp_score)
    auc = roc_auc_score(sp_true_presence, sp_score)
    mcc = matthews_corrcoef(sp_true_presence, y_pred_thresholded)

    val_metrics = {'loss': total_loss / len(valid_data), 'AUC': auc, 'AUPRC' : prc, 'MCC': mcc}
    #matthews_corrcoef(y_true, y_pred)
    return (total_loss / len(valid_data)), val_metrics


def main_training_loop(args: argparse.ArgumentParser):
    if args.enforce_walltime == True:
        loop_start_time = time.time()
        logger.info('Started timing loop')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #Setup Model
    #TODO how to inject new num_labels here?
    logger.info(f'Loading pretrained model in {args.resume}')
    config = ProteinAWDLSTMConfig.from_pretrained(args.resume)
    #override old model config from commandline args
    setattr(config, 'num_labels', args.num_labels)
    setattr(config, 'classifier_hidden_size', args.classifier_hidden_size)
    model = ProteinAWDLSTMCRF.from_pretrained(args.resume, config = config)    
    #training logger
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    experiment_name = f"{args.experiment_name}_{model.base_model_prefix}_{time_stamp}"
    viz = visualization.get(args.output_dir, experiment_name, local_rank = -1) #debug=args.debug) #this -1 means traning is not distributed, debug makes experiment dry run for wandb

    train_data = ThreeLineFastaDataset(os.path.join(args.data, 'train.fasta'))
    val_data = ThreeLineFastaDataset(os.path.join(args.data, 'valid.fasta'))


    train_loader = DataLoader(train_data, batch_size =args.batch_size, collate_fn= train_data.collate_fn, shuffle = True)
    val_loader = DataLoader(val_data, batch_size =args.batch_size, collate_fn= train_data.collate_fn)

    logger.info(f'Data loaded. One epoch = {len(train_loader)} batches.')

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    
    model.to(device)
    logger.info('Model set up!')
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model has {num_parameters} trainable parameters')

    if torch.cuda.is_available():
        model, optimizer = amp.initialize(model, optimizer, opt_level='O0')#'O1')
    else :
        logger.info(f'Running model on {device}, not using nvidia apex')

    #set up wandb logging, tape visualizer class takes care of everything. just login to wandb in the env as usual
    viz.log_config(args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)
    logger.info(f'Logging experiment as {experiment_name} to wandb/tensorboard')
        
    #keep track of best loss
    stored_loss = 100000000
    learning_rate_steps = 0

    global_step = 0
    for epoch in range(1, args.epochs+1):
        logger.info(f'Starting epoch {epoch}')
        viz.log_metrics({'Learning Rate': optimizer.param_groups[0]['lr'] }, "train", global_step)

        epoch_start_time = time.time()
        start_time = time.time() #for lr update interval
        
        for i, batch in enumerate(train_loader):

            data, targets = batch
            loss = training_step(model, data, targets, optimizer, args, i)
            viz.log_metrics({'loss': loss}, "train", global_step)
            global_step += 1
            #logger.info(f'Minibatch {i}/{len(train_loader)} processed. Shape {data.shape}. Memory allocated {torch.cuda.memory_allocated(device)}')

        logger.info(f'Step {global_step}, Epoch {epoch}: validating for {len(val_loader)} Validation steps')
        val_loss, val_metrics = validate(model, val_loader)
        #val_metrics = {'loss': val_loss}
        viz.log_metrics(val_metrics, "val", global_step)


        if val_loss < stored_loss:
            model.save_pretrained(args.output_dir)
            #also save with apex
            if torch.cuda.is_available():
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict()
                    }
                torch.save(checkpoint, os.path.join(args.output_dir, 'amp_checkpoint.pt'))
                logger.info(f'New best model with loss {loss}, Saving model, training step {global_step}')
            stored_loss = loss

        if  args.enforce_walltime == True and (time.time() - loop_start_time) > 84600: #23.5 hours
            logger.info('Wall time limit reached, ending training early')
            return val_loss



        logger.info(f'Epoch {epoch} training complete')
        logger.info(f'Epoch {epoch}, took {time.time() - epoch_start_time:.2f}.\t Train loss: {loss:.2f}')

    return val_loss



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='AWD-LSTM language modeling')
    parser.add_argument('--data', type=str, default='../data/data/signalp_5_data/',
                        help='location of the data corpus. Expects test, train and valid .txt')

    #args relating to training strategy.
    parser.add_argument('--lr', type=float, default=10,
                        help='initial learning rate')
    parser.add_argument('--lr_step', type = float, default = 0.9,
                        help = 'factor by which to multiply learning rate at each reduction step')
    parser.add_argument('--update_lr_steps', type = int, default = 6000,
                        help = 'After how many update steps to check for learning rate update')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--wait_epochs', type = int, default = 3,
                        help='Reduce learning rates after wait_epochs epochs without improvement')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--optimizer', type=str,  default='sgd',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--reset_hidden', type=bool, default=False,
                        help = 'Reset the hidden state after encounter of the tokenizer stop token')
    parser.add_argument('--log_interval', type=int, default=10000, metavar='N',
                        help='report interval')
    parser.add_argument('--output_dir', type=str,  default=f'{time.strftime("%Y-%m-%d",time.gmtime())}_awd_lstm_lm_pretraining',
                        help='path to save logs and trained model')
    parser.add_argument('--wandb_sweep', type=bool, default=False,
                        help='wandb hyperparameter sweep: Override hyperparams with params from wandb')
    parser.add_argument('--resume', type=str,  default='',
                        help='path of model to resume (directory containing .bin and config.json')
    parser.add_argument('--experiment_name', type=str,  default='AWD_LSTM_LM',
                        help='experiment name for logging')
    parser.add_argument('--enforce_walltime', type=bool, default =True,
                        help='Report back current result before 24h wall time is over')


    #args for model architecture
    parser.add_argument('--classifier_hidden_size', type=int, default=128, metavar='N',
                        help='Hidden size of the classifier head MLP')
    parser.add_argument('--num_labels', type=int, default=6, metavar='N',
                        help='Number of labels for the classifier head')


    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    #make unique output dir
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    args.output_dir = os.path.join(args.output_dir, args.experiment_name+time_stamp)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)


    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%y/%m/%d %H:%M:%S")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    #choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Running on: {device}')
    logger.info(f'Saving to {args.output_dir}')
    main_training_loop(args)