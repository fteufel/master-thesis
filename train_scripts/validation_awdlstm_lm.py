# Script to run and time the AWD-LSTM validation set.
# Data is concatenated to one long sequence, out of which minibatches are cut (truncated backpropagatation through time)
# Adapted to work with Huggingface-style AWD-LSTM, apart from that still not very modular
#to debug validation
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
from tape import TAPETokenizer
from utils.visualization import get
import utils.visualization as visualization
from training_utils import repackage_hidden, save_training_status
from training_utils import VirtualBatchTruncatedBPTTHdf5Dataset as Hdf5Dataset
from torch.utils.data import DataLoader
from apex import amp

import data
import os 
import random
import hashlib


def validate(model: torch.nn.Module, valid_data: DataLoader , optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    '''Run over the validation data. Average loss over the full set.
    '''
    model.eval()

    total_loss = 0
    total_len = 0
    for i, batch in enumerate(valid_data):
        data, targets = batch
        #data = data.to(device)
        #targets = targets.to(device)
        #seq_len = len(data)

        #if i == 0:
        #    loss, output, hidden = model(data, targets= targets)
        #else:
        #    loss, output, hidden = model(data, hidden_state = hidden, targets = targets)

        #scaled_loss = loss.item() * seq_len
        #total_loss += scaled_loss #scale by length

        #total_len += seq_len
        #hidden = repackage_hidden(hidden) #detach from graph
        if i % 500 == 0:
            logger.info(f'Processed {i} minibatches.')

    return 10000#total_loss / total_len #normalize by seq len again


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
    viz = get(args.output_dir, experiment_name, local_rank = -1) #debug=args.debug) #this -1 means traning is not distributed, debug makes experiment dry run for wandb

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
    
    model.to(device)
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
    logger.info('loading data')
    valid_data = Hdf5Dataset(os.path.join(args.data, 'valid.hdf5'), batch_size= args.batch_size, bptt_length= args.bptt, buffer_size=args.buffer_size)
    valid_loader = DataLoader(valid_data, batch_size =1, collate_fn= valid_data.collate_fn)
    epoch_start_time = time.time()
    logger.info(f'starting validation. Will take {len(valid_data)} steps')
    val_loss = validate(model, valid_loader, optimizer, args)
    val_metrics = {'loss': val_loss, 'perplexity': math.exp(val_loss)}
    viz.log_metrics(val_metrics, "val", global_step)
    logger.info(f'Validation set complete. Full set took {time.time()-epoch_start_time} seconds')
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
    parser.add_argument('--buffer_size', type= int, default = 5000000,
                        help = 'How much data to load into RAM (in bytes')
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

    main_training_loop(args)
