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
from models.awd_lstm import ProteinAWDLSTMForLM, ProteinAWDLSTMConfig
from tape import TAPETokenizer, visualization
from training_utils import TruncatedBPTTDataset, repackage_hidden
from torch.utils.data import DataLoader

import data
import os 
import random
import hashlib


def train_epoch(model: torch.nn.Module, train_data: DataLoader , optimizer: torch.optim.Optimizer, args: argparse.ArgumentParser, global_step :int, visualizer: visualization.TAPEVisualizer):
    '''
    better training function, things are passed explicitly here
    Trains for one epoch
    model : model to train
    train_data: train data as DataLoader
    global_step: cross-epoch global step for logging
    Important: Learning rate scaling setup only works when minibatches have seq_len as dim 1 
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

        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        visualizer.log_metrics({'loss': loss.item()}, "train", global_step)
        total_loss += loss.item()
        #reset learning rate
        optimizer.param_groups[0]['lr'] = lr2

        logger.info(f'Processed {total_len} sequence tokens.')
        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | ' #TODO remnant from original awd-lstm repo
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                0, i, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()

        global_step += 1
        total_len += seq_len

    print('epoch complete')
    return global_step, cur_loss #arbitrary choice


def validate(model: torch.nn.Module, valid_data: DataLoader , optimizer: torch.optim.Optimizer, args: argparse.ArgumentParser):
    '''
    Run the validation data. Average loss over the full set.
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

        #TODO detach from graph, maybe .data does the trick (deprecated who cares)
        scaled_loss = loss.item() * seq_len
        total_loss += scaled_loss #scale by length

        total_len += seq_len
        hidden = repackage_hidden(hidden) #detach from graph

    return total_loss / total_len #normalize by seq len again


def main_training_loop(args: argparse.ArgumentParser):

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    tokenizer = TAPETokenizer(vocab = 'iupac')


    #TODO don't see why I need to differentiate here
    eval_batch_size = args.batch_size
    #test_batch_size = 1
    train_data = TruncatedBPTTDataset(os.path.join(args.data, 'train.txt'), tokenizer, args.batch_size, args.bptt)
    valid_data = TruncatedBPTTDataset(os.path.join(args.data, 'valid.txt'), tokenizer, eval_batch_size, args.bptt)
    #Not testing before I have my hyperparams. test_data  = TruncatedBPTTDataset(os.path.join(args.data, 'test.txt'), tokenizer, test_batch_size, args.bptt)

    train_loader = DataLoader(train_data, batch_size =1, collate_fn= train_data.collate_fn)
    valid_loader = DataLoader(valid_data, batch_size =1, collate_fn= valid_data.collate_fn)
    #test_loader = DataLoader(test_data, batch_size =1, collate_fn= test_data.collate_fn)

    #setup model
    config = ProteinAWDLSTMConfig()
    config.input_size = tokenizer.vocab_size
    if args.reset_hidden:
        logger.info(f'Resetting hidden state after {tokenizer.stop_token}')
        config.reset_token_id = tokenizer.convert_token_to_id(tokenizer.stop_token)

    model = ProteinAWDLSTMForLM(config)

    if args.resume:
        logger.info(f'Loading pretrained model in {args.resume}')
        model = ProteinAWDLSTMForLM.from_pretrained(args.resume)
    
    if args.wandb_sweep:
        #Set all the params to those brought from wandb config
        logger.info(f'Receiving config from wandb!')
        import wandb
        from training_utils import override_from_wandb
        override_from_wandb(wandb.config, ars, config)
        model = ProteinAWDLSTMForLM(config)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    
    logger.info('Model set up!')
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model has {num_parameters} trainable parameters')

    #set up wandb logging, tape visualizer class takes care of everything. just login to wandb in the env as usual
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    experiment_name = f"{'tbptt_language_modeling'}_{model.base_model_prefix}_{time_stamp}_{random.randint(0, int(1e6)):0>6d}"
    viz = visualization.get(args.output_dir, experiment_name, local_rank = -1) #debug=args.debug) #this -1 means traning is not distributed, debug makes experiment dry run for wandb
    viz.log_config(args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)
    logger.info(f'Logging experiment as {experiment_name} to wandb/tensorflow')
        


    #keep track of best loss
    best_val_loss = None
    num_epochs_no_improvement = 0
    stored_loss = 100000000
    learning_rate_steps = 0

    global_step = 0
    for epoch in range(1, args.epochs+1):
        viz.log_metrics({'Learning Rate': optimizer.param_groups[0]['lr'] }, "train", global_step)
        epoch_start_time = time.time()
        #train
        global_step, train_loss = train_epoch(model, train_loader, optimizer, args, global_step = global_step, visualizer = viz)
        #eval
        val_loss = validate(model, valid_loader, optimizer, args)

        logger.info(f'Epoch{epoch}, took {time.time() - epoch_start_time}.\n Val loss: {val_loss} \t Val perplexity: {math.exp(val_loss)}')
        val_metrics = {'loss': val_loss, 'perplexity': math.exp(val_loss)}
        viz.log_metrics(val_metrics, "val", global_step)


        if val_loss < stored_loss:
            model.save_pretrained(args.output_dir)
            print('Saving model (new best validation)')
        else:
            num_epochs_no_improvement += 1
        
        if num_epochs_no_improvement == args.wait_epochs:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * args.lr_step
            learning_rate_steps += 1

        
        if learning_rate_steps > 5:
            break
            #TODO when do I save? best model, or last model>
            #obviously best model is best, but how does everyone else do it



if __name__ == '__main__':
    #removed all model architecture specific CLI args. This is supported via a JSON config file.
    parser = argparse.ArgumentParser(description='AWD-LSTM language modeling')
    parser.add_argument('--data', type=str, default='data/awdlstmtestdata/',
                        help='location of the data corpus')

    #args relating to training strategy. Eventually, rename to TAPE names
    parser.add_argument('--lr', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--optimizer', type=str,  default='sgd',
                        help='optimizer to use (sgd, adam)')
    #parser.add_argument('--when', nargs="+", type=int, default=[-1],
    #                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
    parser.add_argument('--wait_epochs', type = int, default = 3,
                        help='Reduce learning rates after wait_epochs epochs without improvement')
    parser.add_argument('--lr_step', type = float, default = 0.9,
                        help = 'factor by which to multiply learning rate at each reduction step')
    parser.add_argument('--reset_hidden', type=bool, default=False,
                        help = 'Reset the hidden state after encounter of the tokenizer stop token')

    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--output_dir', type=str,  default=f'{time.strftime("%Y-%m-%d",time.gmtime())}_awd_lstm_lm_pretraining',
                        help='path to save logs and trained model')
    #parser.add_argument('--debug', type=bool, default=False,
    #                    help='Control whether to log to w and b or not')
    parser.add_argument('--wandb-sweep', type=bool, default=False,
                        help='wandb hyperparameter sweep: Override hyperparams with params from wandb')


    parser.add_argument('--resume', type=str,  default='',
                        help='path of model to resume (directory containing .bin and config.json')

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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Running on: {device}')

    main_training_loop(args)