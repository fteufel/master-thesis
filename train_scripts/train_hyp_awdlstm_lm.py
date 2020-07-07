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
from typing import Tuple
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet
from models.hyperbolic_awd_lstm import ProteinAWDLSTMForLM, ProteinAWDLSTMConfig
from tape import TAPETokenizer
from tape import visualization #import utils.visualization as visualization
from training_utils import repackage_hidden, save_training_status
from training_utils import VirtualBatchTruncatedBPTTHdf5Dataset as Hdf5Dataset
from torch.utils.data import DataLoader

import data
import os 
import random
import hashlib

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


def training_step(model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, previous_hidden: tuple, optimizer: torch.optim.Optimizer, 
                    args: argparse.ArgumentParser, i: int) -> (float, tuple):
    '''Predict one minibatch and performs update step.
    Returns:
        loss: loss value of the minibatch
        hidden: final hidden state to reuse for next minibatch    
    '''
    data = data.to(device)
    targets = targets.to(device)

    #scale learning rate
    seq_len = len(data)

    lr2 = optimizer.param_groups[0]['lr']
    optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt #divide to normalize
    model.train()

    optimizer.zero_grad()

    if i == 0:
        loss, output, hidden = model(data, targets= targets)
    else:
        hidden = repackage_hidden(previous_hidden) #detaches hidden state from graph of previous step
        loss, output, hidden = model(data, hidden_state = hidden, targets = targets)

    loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    if args.clip: 
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()
    optimizer.param_groups[0]['lr'] = lr2

    return loss.item(), hidden


def validate(model: torch.nn.Module, valid_data: DataLoader , optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> float:
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
            loss, output, hidden = model(data, targets= targets)
        else:
            loss, output, hidden = model(data, hidden_state = hidden, targets = targets)

        scaled_loss = loss.item() * seq_len
        total_loss += scaled_loss #scale by length

        total_len += seq_len
        hidden = repackage_hidden(hidden) #detach from graph

    return total_loss / total_len #normalize by seq len again

def main_training_loop(args: argparse.ArgumentParser):
    if args.enforce_walltime == True:
        loop_start_time = time.time()
        logger.info('Started timing loop')

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

    train_data = Hdf5Dataset(os.path.join(args.data, 'train.hdf5'), batch_size= args.batch_size, bptt_length= args.bptt, buffer_size=args.buffer_size)
    val_data = Hdf5Dataset(os.path.join(args.data, 'valid.hdf5'), batch_size= args.batch_size, bptt_length= args.bptt, buffer_size=args.buffer_size)

    logger.info(f'Data loaded. One epoch = {len(train_data)} steps.')

    train_loader = DataLoader(train_data, batch_size =1, collate_fn= train_data.collate_fn)
    val_loader = DataLoader(val_data, batch_size =1, collate_fn= train_data.collate_fn)

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
    logger.info(f'Running model on {device}')

    #set up wandb logging, tape visualizer class takes care of everything. just login to wandb in the env as usual
    viz.log_config(args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)
    logger.info(f'Logging experiment as {experiment_name} to wandb/tensorboard')
        
    #keep track of best loss
    num_epochs_no_improvement = 0
    stored_loss = 100000000
    learning_rate_steps = 0

    global_step = 0
    for epoch in range(1, args.epochs+1):
        logger.info(f'Starting epoch {epoch}')
        viz.log_metrics({'Learning Rate': optimizer.param_groups[0]['lr'] }, "train", global_step)

        epoch_start_time = time.time()
        start_time = time.time() #for lr update interval
        hidden = None
        for i, batch in enumerate(train_loader):

            data, targets = batch
            loss, hidden = training_step(model, data, targets, hidden, optimizer, args, i)
            viz.log_metrics({'loss': loss, 'perplexity': math.exp(loss)}, "train", global_step)
            global_step += 1

            # every update_lr_steps, evaluate performance and save model/progress in learning rate
            if global_step % args.update_lr_steps == 0 and global_step > 0:

                val_loss = validate(model, val_loader, optimizer, args)
                val_metrics = {'loss': val_loss, 'perplexity': math.exp(val_loss)}
                viz.log_metrics(val_metrics, "val", global_step)

                elapsed = time.time() - start_time
                logger.info(f'Training step {global_step}, { elapsed / args.log_interval:.3f} s/batch. tr_loss: {loss:.2f}, tr_perplexity {math.exp(loss):.2f} va_loss: {val_loss:.2f}, va_perplexity {math.exp(val_loss):.2f}')
                start_time = time.time()

                if val_loss < stored_loss:
                    model.save_pretrained(args.output_dir)
                    save_training_status(args.output_dir, epoch, global_step, num_epochs_no_improvement, stored_loss, learning_rate_steps)
                    logger.info(f'New best model with loss {loss}, Saving model, training step {global_step}')
                    stored_loss = loss
                else:
                    num_epochs_no_improvement += 1
                    logger.info(f'Step {global_step}: No improvement for {num_epochs_no_improvement} pseudo-epochs.')

                    if num_epochs_no_improvement == args.wait_epochs:
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * args.lr_step
                        learning_rate_steps += 1
                        num_epochs_no_improvement = 0
                        logger.info(f'Step {global_step}: Decreasing learning rate. learning rate step {learning_rate_steps}.')
                        viz.log_metrics({'Learning Rate': optimizer.param_groups[0]['lr'] }, "train", global_step)

                        #break early after 5 lr steps
                        if learning_rate_steps > 5:
                            logger.info('Learning rate step limit reached, ending training early')
                            return val_loss


            if  args.enforce_walltime == True and (time.time() - epoch_start_time) > 84600: #23.5 hours
                logger.info('Wall time limit reached, ending training early')
                return val_loss



        logger.info(f'Epoch {epoch} training complete')
        logger.info(f'Epoch {epoch}, took {time.time() - epoch_start_time:.2f}.\t Train loss: {loss:.2f} \t Train perplexity: {math.exp(loss):.2f}')

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
    parser.add_argument('--enforce_walltime', type=bool, default =True,
                        help='Report back current result before 24h wall time is over')
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
    logger.info(f'Saving to {args.output_dir}')

    main_training_loop(args)