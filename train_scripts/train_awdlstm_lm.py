# Script to run AWD-LSTM pretraining style language modeling
# Data is concatenated to one long sequence, out of which minibatches are cut (truncated backpropagatation through time)
# Adapted to work with Huggingface-style AWD-LSTM, apart from that still not very modular
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
from models.awd_lstm import ProteinAWDLSTMForLM, ProteinAWDLSTMConfig
from tape import TAPETokenizer, visualization
from training_utils import repackage_hidden
from training_utils import VirtualBatchTruncatedBPTTHdf5Dataset as Hdf5Dataset
from torch.utils.data import DataLoader
from apex import amp

import data
import os 
import random
import hashlib


def train_epoch(model: torch.nn.Module, train_data: DataLoader , optimizer: torch.optim.Optimizer, args: argparse.ArgumentParser, global_step :int, visualizer: visualization.TAPEVisualizer):
    '''Trains model for one epoch.
    Args:
        model : model to train
        train_data: train data as DataLoader
        global_step: cross-epoch global step for logging
    Important: Learning rate scaling setup only works when minibatches have seq_len as dim 0 
    '''

    total_loss = 0
    start_time = time.time()
    #batch, i = 0, 0
    total_len = 0
    cur_loss = None
    for i, batch in enumerate(train_data):

        data, targets = batch
        data.to(device)
        targets.to(device)

        #scale learning rate
        seq_len = len(data)
        #print(f' Training loop: Seq len check: {seq_len}')
        #print(data.shape)
        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt #divide to normalize
        model.train()

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()

        if i == 0:
            loss, output, hidden = model(data, targets= targets)
        else:
            hidden = repackage_hidden(hidden)
            loss, output, hidden = model(data, hidden_state = hidden, targets = targets)

        if torch.cuda.is_available():
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        visualizer.log_metrics({'loss': loss.item(), 'perplexity': math.exp(loss.item())}, "train", global_step)
        total_loss += loss.item()
        #reset learning rate
        optimizer.param_groups[0]['lr'] = lr2

        if global_step % args.log_interval == 0 and global_step > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logger.info(f'Training step {global_step}, { elapsed / args.log_interval:.3f} s/batch. loss: {cur_loss:.2f}, perplexity {math.exp(cur_loss):.2f}')
            total_loss = 0
            start_time = time.time()

        global_step += 1
        total_len += seq_len

    return global_step, cur_loss #arbitrary choice


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

        data, targets = batch
        data.to(device)
        targets.to(device)

        #scale learning rate
        seq_len = len(data)
        #print(f' Training loop: Seq len check: {seq_len}')
        #print(data.shape)
        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt #divide to normalize
        model.train()

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()

        if i == 0:
            loss, output, hidden = model(data, targets= targets)
        else:
            hidden = repackage_hidden(hidden)
            loss, output, hidden = model(data, hidden_state = hidden, targets = targets)

        if torch.cuda.is_available():
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        visualizer.log_metrics({'loss': loss.item(), 'perplexity': math.exp(loss.item())}, "train", global_step)
        total_loss += loss.item()
        #reset learning rate
        optimizer.param_groups[0]['lr'] = lr2

        if global_step % args.log_interval == 0 and global_step > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logger.info(f'Training step {global_step}, { elapsed / args.log_interval:.3f} s/batch. loss: {cur_loss:.2f}, perplexity {math.exp(cur_loss):.2f}')
            total_loss = 0
            start_time = time.time()
        
        if global_step % args.update_lr_steps == 0 and global_step > 0:
            if loss.item() < stored_loss:
                model.save_pretrained(args.output_dir)
                #also save with apex
                if torch.cuda.is_available():
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                        }
                    torch.save(checkpoint, os.path.join(args.output_dir, 'amp_checkpoint.pt'))
                    print('Saving model (new best validation)')
                stored_loss = loss.item()
            else:
                num_epochs_no_improvement += 1
            
            if num_epochs_no_improvement == args.wait_epochs:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * args.lr_step
                learning_rate_steps += 1

        global_step += 1
        total_len += seq_len

    return global_step, cur_loss #arbitrary choice


def validate(model: torch.nn.Module, valid_data: DataLoader , optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    '''Run over the validation data. Average loss over the full set.
    '''
    model.eval()

    total_loss = 0
    total_len = 0
    for i, batch in enumerate(valid_data):
        data, targets = batch
        data.to(device)
        targets.to(device)
        seq_len = len(data)

        if i == 0:
            loss, output, hidden = model(data, targets= targets)
        else:
            loss, output, hidden = model(data, hidden_state = hidden, targets = targets)

        scaled_loss = loss.item() * seq_len
        total_loss += scaled_loss #scale by length

        total_len += seq_len
        hidden = repackage_hidden(hidden) #detach from graph

    return total_loss / total_len #normalize by seq len again


def main_training_loop(args: argparse.ArgumentParser):

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #Setup Model
    tokenizer = TAPETokenizer(vocab = 'iupac') 
    config = ProteinAWDLSTMConfig(**vars(args))
    config.vocab_size = tokenizer.vocab_size
    if args.reset_hidden:
        logger.info(f'Resetting hidden state after {tokenizer.stop_token}')
        config.reset_token_id = tokenizer.convert_token_to_id(tokenizer.stop_token)

    model = ProteinAWDLSTMForLM(config)
    #training logger
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    experiment_name = f"{args.experiment_name}_{model.base_model_prefix}_{time_stamp}"
    viz = visualization.get(args.output_dir, experiment_name, local_rank = -1) #debug=args.debug) #this -1 means traning is not distributed, debug makes experiment dry run for wandb


    #deprecated, new dataset can handle 1D hdf5. # hdf5 files with correct batch size need to be created ahead of training (takes 100gb ram and 2 hours)
    #logger.info('Loading validation data into memory...')
    #valid_data = TruncatedBPTTHdf5Dataset(os.path.join(args.data, f'valid_{args.batch_size}.hdf5'), args.bptt)
    #logger.info('Loading training data into memory...')
    #train_data = TruncatedBPTTHdf5Dataset(os.path.join(args.data, f'train_{args.batch_size}.hdf5'), args.bptt)
    
    #Not testing before I have my hyperparams. test_data  = TruncatedBPTTDataset(os.path.join(args.data, 'test.txt'), tokenizer, test_batch_size, args.bptt)
    train_data = Hdf5Dataset(os.path.join(args.data, 'train.hdf5'), batch_size= args.batch_size, bptt_length= args.bptt, buffer_size=args.buffer_size)
    logger.info(f'Data loaded. One epoch = {len(train_data)} steps.')

    train_loader = DataLoader(train_data, batch_size =1, collate_fn= train_data.collate_fn)

    #test_loader = DataLoader(test_data, batch_size =1, collate_fn= test_data.collate_fn)


    #overwrite model when restarting/changing params
    if args.resume:
        logger.info(f'Loading pretrained model in {args.resume}')
        model = ProteinAWDLSTMForLM.from_pretrained(args.resume)
    
    if args.wandb_sweep:
        #This prevents errors. When model is partly set up from config, not commmandline,
        #config that is received from wandb might not match what is in args and ProteinConfig.
        #when then calling log_config, inconsistency would throw an error.
        #this overwrites args and ProteinConfig, so wandb has priority.
        #In case of doubt of match, check wandb run and save_pretrained config.json. Should always agree
        logger.info(f'Receiving config from wandb!')
        import wandb
        from training_utils import override_from_wandb
        override_from_wandb(wandb.config, args, config)
        model = ProteinAWDLSTMForLM(config)
    

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    
    logger.info('Model set up!')
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model has {num_parameters} trainable parameters')

    if torch.cuda.is_available():
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    else :
        logger.info(f'Running model on {device}, not using nvidia apex')

    #set up wandb logging, tape visualizer class takes care of everything. just login to wandb in the env as usual
    viz.log_config(args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)
    logger.info(f'Logging experiment as {experiment_name} to wandb/tensorflow')
        

    #keep track of best loss
    num_epochs_no_improvement = 0
    stored_loss = 100000000
    learning_rate_steps = 0

    global_step = 0
    for epoch in range(1, args.epochs+1):
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


        #if train_loss < stored_loss: #done in epoch loop
        #    model.save_pretrained(args.output_dir)
            #also save with apex
        #    checkpoint = {
        #        'model': model.state_dict(),
        #        'optimizer': optimizer.state_dict(),
        #        'amp': amp.state_dict()
        #        }
        #    torch.save(checkpoint, os.path.join(args.output_dir, 'amp_checkpoint.pt'))
        #    print('Saving model (new best validation)')
        #    stored_loss = train_loss
        #else:
        #    num_epochs_no_improvement += 1
        
        #if num_epochs_no_improvement == args.wait_epochs:
        #    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * args.lr_step
        #    learning_rate_steps += 1

        #last epoch - get validation performance
        if learning_rate_steps > 5:
            valid_data = Hdf5Dataset(os.path.join(args.data, 'valid.hdf5'), batch_size= args.batch_size, bptt_length= args.bptt, buffer_size=args.buffer_size)
            valid_loader = DataLoader(valid_data, batch_size =1, collate_fn= valid_data.collate_fn)
            val_loss = validate(model, valid_loader, optimizer, args)
            val_metrics = {'loss': val_loss, 'perplexity': math.exp(val_loss)}
            viz.log_metrics(val_metrics, "val", global_step)

            return val_loss

    valid_data = Hdf5Dataset(os.path.join(args.data, 'valid.hdf5'), batch_size= args.batch_size, bptt_length= args.bptt, buffer_size=args.buffer_size)
    valid_loader = DataLoader(valid_data, batch_size =1, collate_fn= valid_data.collate_fn)
    val_loss = validate(model, valid_loader, optimizer, args)
    val_metrics = {'loss': val_loss, 'perplexity': math.exp(val_loss)}
    viz.log_metrics(val_metrics, "val", global_step)    
    
    return val_loss


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='AWD-LSTM language modeling')
    parser.add_argument('--data', type=str, default='../data/awdlstmtestdata/',
                        help='location of the data corpus. Expects test, train and valid .txt')

    #args relating to training strategy.
    parser.add_argument('--lr', type=float, default=10,
                        help='initial learning rate')
    parser.add_argument('--lr_step', type = float, default = 0.9,
                        help = 'factor by which to multiply learning rate at each reduction step')
    parser.add_argument('--update_lr_steps', type = int, default = 60000,
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
    parser.add_argument('--buffer_size', type= int, default = 5000000,
                        help = 'How much data to load into RAM (in bytes')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--output_dir', type=str,  default=f'{time.strftime("%Y-%m-%d",time.gmtime())}_awd_lstm_lm_pretraining',
                        help='path to save logs and trained model')
    parser.add_argument('--wandb_sweep', type=bool, default=False,
                        help='wandb hyperparameter sweep: Override hyperparams with params from wandb')
    parser.add_argument('--resume', type=str,  default='',
                        help='path of model to resume (directory containing .bin and config.json')
    parser.add_argument('--experiment_name', type=str,  default='AWD_LSTM_LM',
                        help='experiment name for logging')
    #args for model architecture
    parser.add_argument('--num_hidden_layers', type=int, default=3, metavar='N',
                        help='report interval')
    parser.add_argument('--input_size', type=int, default=10, metavar='N',
                        help='Embedding layer size')
    parser.add_argument('--hidden_size', type=int, default=1000, metavar='N',
                        help='LSTM hidden size')
    parser.add_argument('--dropout_prob', type=float, default=0.4,
                        help='Dropout')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.3,
                        help='Dropout between layers')
    parser.add_argument('--embedding_dropout_prob', type=float, default=0.1,
                        help='Dropout embedding layer')
    parser.add_argument('--input_dropout_prob', type=float, default=0.65,
                        help='Dropout input')
    parser.add_argument('--weight_dropout_prob', type=float, default=0.5,
                        help='Dropout LSTM weights')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Activation regularization beta')
    parser.add_argument('--alpha', type=float, default=2.0,
                        help='Activation regularization alpha')


    args = parser.parse_args()


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

    main_training_loop(args)
