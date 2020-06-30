#script to resume a training run from its last checkpoint
#necessary because wall time is 24h, not enough to finish a run
#NOTE cannot start a run from correct batch position yet, restarts epoch
# A bit chaotic because needs to load multiple things:
# - reload model weights with model.from_pretrained()
# - reload training loop hyperparameters from w&b
# - reload apex amp state
# - reload last training loop status from json 

import argparse
import time
import math
import numpy as np
import torch
import logging
import torch.nn as nn
import sys
sys.path.append("..")
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet
from models.sha_rnn import ProteinSHARNNForLM, ProteinSHARNNConfig
from utils.lamb import Lamb
from tape import TAPETokenizer
from training_utils import repackage_hidden, load_training_status
from training_utils import VirtualBatchTruncatedBPTTHdf5Dataset as Hdf5Dataset
from utils.visualization import ResumeWandBVisualizer
from torch.utils.data import DataLoader
from apex import amp

import data
import os 
import random
import hashlib


def train_pseudo_epoch(model: torch.nn.Module, train_data: DataLoader , optimizer: torch.optim.Optimizer, args: argparse.ArgumentParser, 
                        global_step :int, visualizer: visualization.TAPEVisualizer,
                        num_epochs_no_improvement: int = 0, stored_loss: float = 10000000, learning_rate_steps: int =0):
    '''Trains model for the given number of update_lr_steps. Then evaluates perplexity, checks for improvement and updates learning rate and saved model.
    Continues until actual data epoch is complete, always performing the update after update_lr_steps
    Args:
        model : model to train
        train_data: train data as DataLoader
        global_step: cross-epoch global step for logging
        num_epochs_no_improvement: Number of elapsed pseudo-epochs without improvement 
        stored_loss: previous best loss
        learning_rate_steps: Number of elapsed lr update steps
    Important: Learning rate scaling setup only works when minibatches have seq_len as dim 0 
    '''

    #keep track of best loss
    num_epochs_no_improvement = 0
    stored_loss = 100000000
    learning_rate_steps = 0

    total_loss = 0
    start_time = time.time()
    #batch, i = 0, 0
    total_len = 0
    cur_loss = None
    for i, batch in enumerate(train_data):
        #warmup
        for param_group in optimizer.param_groups:
            step = global_step
            pctwarm = min(step, wandb.config.warmup) / wandb.config.warmup

            param_group['lr'] = wandb.config.lr * (pctwarm)

        data, targets = batch
        data = data.to(device)
        targets = targets.to(device)

        #scale learning rate
        seq_len = len(data)
        #print(f' Training loop: Seq len check: {seq_len}')
        #print(data.shape)
        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / wandb.config.bptt #divide to normalize
        model.train()

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()

        if i == 0:
            loss, output, hidden, memory = model(data, targets= targets)
        else:
            hidden = repackage_hidden(hidden)
            memory = repackage_hidden(memory)
            loss, output, hidden, memory = model(data, hidden_state = hidden, memory = memory, targets = targets)

        if torch.cuda.is_available():
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if wandb.config.clip: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), wandb.config.clip)
        optimizer.step()

        visualizer.log_metrics({'loss': loss.item(), 'perplexity': math.exp(loss.item())}, "train", global_step)
        total_loss += loss.item()
        #reset learning rate
        optimizer.param_groups[0]['lr'] = lr2

        if global_step % wandb.config.log_interval == 0 and global_step > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logger.info(f'Training step {global_step}, { elapsed / wandb.config.log_interval:.3f} s/batch. loss: {cur_loss:.2f}, perplexity {math.exp(cur_loss):.2f}')
            total_loss = 0
            start_time = time.time()
        
        if global_step % wandb.config.update_lr_steps == 0 and global_step > 0:
            if loss.item() < stored_loss:
                model.save_pretrained(wandb.config.output_dir)
                #also save with apex
                if torch.cuda.is_available():
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                        }
                    torch.save(checkpoint, os.path.join(wandb.config.output_dir, 'amp_checkpoint.pt'))
                    logger.info(f'Saving model, training step {global_step}')
                stored_loss = loss.item()
            else:
                num_epochs_no_improvement += 1
            
            if num_epochs_no_improvement == wandb.config.wait_epochs:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * wandb.config.lr_step
                learning_rate_steps += 1

        global_step += 1
        total_len += seq_len

    return global_step, cur_loss, num_epochs_no_improvement, stored_loss, learning_rate_steps #arbitrary choice


def validate(model: torch.nn.Module, valid_data: DataLoader , optimizer: torch.optim.Optimizer):
    '''Run over the validation data. Average loss over the full set.
    '''
    model.eval()

    total_loss = 0
    total_len = 0
    for i, batch in enumerate(valid_data):
        data, targets = batch
        data = data.to(device)
        targets = targets.to(device)
        seq_len = len(data)

        if i == 0:
            loss, output, hidden, memory = model(data, targets= targets)
        else:
            loss, output, hidden, memory = model(data, hidden_state = hidden, memory = memory, targets = targets)

        scaled_loss = loss.item() * seq_len
        total_loss += scaled_loss #scale by length

        total_len += seq_len
        hidden = repackage_hidden(hidden) #detach from graph
        memory = repackage_hidden(memory)

    return total_loss / total_len #normalize by seq len again


def main_training_loop(args: argparse.ArgumentParser):


    #load pretrained model
    model = ProteinSHARNNForLM.from_pretrained(args.output_dir)

    #resume training status and reconnect to WandB run
    training_status = load_training_status(args.output_dir)
    viz = ResumeWandBVisualizer(args.output_dir, name = training_status["wandb_name"])

    train_data = Hdf5Dataset(os.path.join(wandb.config.data, 'train.hdf5'), batch_size= wandb.config.batch_size, bptt_length= wandb.config.bptt, buffer_size=wandb.config.buffer_size)
    logger.info(f'Data loaded. One epoch = {len(train_data)} steps.')

    train_loader = DataLoader(train_data, batch_size =1, collate_fn= train_data.collate_fn)



    if wandb.config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.wdecay)
    if wandb.config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.wdecay)
    if wandb.config.optimizer == 'lamb':
        optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wdecay, min_trust=0.25)
    
    model.to(device)
    logger.info('Model set up!')
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model has {num_parameters} trainable parameters')

    if torch.cuda.is_available():
        checkpoint = torch.load(os.path.join(args.output_dir,'amp_checkpoint.pt'))

        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        #NOTE don't understand apex completely yet, is model weight reloading here really necessary?
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        amp.load_state_dict(checkpoint['amp'])
    else :
        logger.info(f'Running model on {device}, not using nvidia apex')

    #set up wandb logging, tape visualizer class takes care of everything. just login to wandb in the env as usual
    viz.watch(model)
    logger.info(f'Logging experiment as {training_status["wandb_name"]} to wandb/tensorflow')


    #keep track of best loss
    num_epochs_no_improvement = training_status['num_epochs_no_improvement']
    stored_loss = training_status['stored_loss']
    learning_rate_steps = training_status['learning_rate_steps']

    global_step = training_status['global_step']
    for epoch in range(training_status['epoch'], wandb.config.epochs+1):
        viz.log_metrics({'Learning Rate': optimizer.param_groups[0]['lr'] }, "train", global_step)
        epoch_start_time = time.time()
        #train
        epoch_output = train_pseudo_epoch(model, train_loader, optimizer, args, global_step = global_step, visualizer = viz, 
                                                        num_epochs_no_improvement=num_epochs_no_improvement, 
                                                        stored_loss = stored_loss, learning_rate_steps=learning_rate_steps)
        #unpack and log
        global_step, train_loss, num_epochs_no_improvement, stored_loss, learning_rate_steps = epoch_output
        logger.info(f'Epoch {epoch} training complete')
        logger.info(f'Epoch {epoch}, took {time.time() - epoch_start_time:.2f}.\t Train loss: {train_loss:.2f} \t Train perplexity: {math.exp(train_loss):.2f}')

        #last epoch - get validation performance
        if learning_rate_steps > 5:
            valid_data = Hdf5Dataset(os.path.join(wandb.config.data, 'valid.hdf5'), batch_size= wandb.config.batch_size, bptt_length= wandb.config.bptt, 
                                                    buffer_size=wandb.config.buffer_size)
            valid_loader = DataLoader(valid_data, batch_size =1, collate_fn= valid_data.collate_fn)
            val_loss = validate(model, valid_loader, optimizer)
            val_metrics = {'loss': val_loss, 'perplexity': math.exp(val_loss)}
            viz.log_metrics(val_metrics, "val", global_step)

            return val_loss

    valid_data = Hdf5Dataset(os.path.join(wandb.config.data, 'valid.hdf5'), batch_size= wandb.config.batch_size, bptt_length= wandb.config.bptt, buffer_size=wandb.config.buffer_size)
    valid_loader = DataLoader(valid_data, batch_size =1, collate_fn= valid_data.collate_fn)
    val_loss = validate(model, valid_loader, optimizer)
    val_metrics = {'loss': val_loss, 'perplexity': math.exp(val_loss)}
    viz.log_metrics(val_metrics, "val", global_step)    
    
    return val_loss


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description='AWD-LSTM language modeling')

    parser.add_argument('--output_dir', type=str,  default=f'{time.strftime("%Y-%m-%d",time.gmtime())}_awd_lstm_lm_pretraining',
                        help='path to save logs and trained model')


    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        raise FileNotFoundError

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

    main_training_loop(args)