# Script to run SHA-RNN pretraining style language modeling
# Data is concatenated to one long sequence, out of which minibatches are cut (truncated backpropagatation through time)
# Basically AWD-LSTM training script, adapted because SHA-RNN also passes memory state in addition to hidden state through bptt batches
#python -u main.py --epochs 32 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --data data/enwik8/ --save ENWIK8.pt --log-interval 10 --seed 5512 
# --optimizer lamb --bptt 1024 --warmup 800 --lr 2e-3 --emsize 1024 --nhid 4096 --nlayers 4 --batch_size 16
# to reproduce paper
import argparse
import time
import math
import numpy as np
import torch
import logging
import torch.nn as nn
import sys
sys.path.append("..")
from models.sha_rnn import ProteinSHARNNForLM, ProteinSHARNNConfig
from sha_rnn.lamb import Lamb
from tape import TAPETokenizer, visualization
from apex import amp
from training_utils import TruncatedBPTTDataset, repackage_hidden
from torch.utils.data import DataLoader

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
    batch, i = 0, 0
    total_len = 0
    cur_loss = None
    for i, batch in enumerate(train_data):

        data, targets = batch
        data.to(device)
        targets.to(device)

        #scale learning rate
        seq_len = len(data)

        #scale learning rate
        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
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
        if args.clip: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        visualizer.log_metrics({'loss': loss.item()}, "train", global_step)
        total_loss += loss.item()
        #reset learning rate
        optimizer.param_groups[0]['lr'] = lr2

        logger.info(f'Processed {total_len} sequence tokens.')
        if global_step % args.log_interval == 0 and global_step > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logger.info(f'Training step {global_step},{ elapsed * 1000 / args.log_interval:.2f} ms/batch. loss: {cur_loss:.2f}, perplexity{math.exp(cur_loss):.2f}')
            total_loss = 0
            start_time = time.time()

        global_step += 1
        total_len += seq_len

    return global_step, cur_loss #arbitrary choice


def validate(model: torch.nn.Module, valid_data: DataLoader , optimizer: torch.optim.Optimizer, args: argparse.ArgumentParser):
    '''
    Run the validation data. Average loss over the full set.
    '''
    model.eval()

    total_loss = 0

    total_loss = 0
    total_len = 0
    for i, batch in enumerate(valid_data):

        data, targets = batch
        data.to(device)
        targets.to(device)
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

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #Setup model
    tokenizer = TAPETokenizer(vocab = 'iupac')
    config = ProteinSHARNNConfig() #from_file, add to args
    config.input_size = tokenizer.vocab_size
    if args.reset_hidden: #does not really work. Attention breaks the causality of this
        logger.info(f'Resetting hidden state after {tokenizer.stop_token}')
        config.reset_token_id = tokenizer.convert_token_to_id(tokenizer.stop_token)

    model = ProteinSHARNNForLM(config)
    #training logger
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    experiment_name = f"{'tbptt_language_modeling'}_{model.base_model_prefix}_{time_stamp}_{random.randint(0, int(1e6)):0>6d}"
    viz = visualization.get(args.output_dir, experiment_name, local_rank = -1, debug=True) #this -1 means traning is not distributed, debug makes experiment dry run for wandb


    #TODO don't see why I need to differentiate here
    eval_batch_size = args.batch_size
    #test_batch_size = 1
    train_data = TruncatedBPTTDataset(os.path.join(args.data, 'train.txt'), tokenizer, args.batch_size, args.bptt)
    valid_data = TruncatedBPTTDataset(os.path.join(args.data, 'valid.txt'), tokenizer, eval_batch_size, args.bptt)
    #Not testing before I have my hyperparams. test_data  = TruncatedBPTTDataset(os.path.join(args.data, 'test.txt'), tokenizer, test_batch_size, args.bptt)

    train_loader = DataLoader(train_data, batch_size =1, collate_fn= train_data.collate_fn)
    valid_loader = DataLoader(valid_data, batch_size =1, collate_fn= valid_data.collate_fn)
    #test_loader = DataLoader(test_data, batch_size =1, collate_fn= test_data.collate_fn)


    #overwrite model when restarting/changing params
    if args.resume:
        logger.info(f'Loading pretrained model in {args.resume}')
        model = ProteinSHARNNForLM.from_pretrained(args.resume)
    
    if args.wandb_sweep:
        #Set all the params to those brought from wandb config
        logger.info(f'Receiving config from wandb!')
        import wandb
        from training_utils import override_from_wandb
        override_from_wandb(wandb.config, args, config)
        model = ProteinSHARNNForLM(config)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'lamb':
        optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wdecay, min_trust=0.25)
    
    logger.info('Model set up!')
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model has {num_parameters} trainable parameters')

    if device == 'cuda:0':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    else :
        logger.info(f'Running model on {torch.device}, not using nvidia apex')

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
        epoch_start_time = time.time()
        #train
        global_step, train_loss = train_epoch(model, train_loader, optimizer, args, global_step = global_step, visualizer = viz)
        logger.info(f'Epoch {epoch} training complete')
        #eval
        val_loss = validate(model, valid_loader, optimizer, args)

        logger.info(f'Epoch{epoch}, took {time.time - epoch_start_time}.\n Val loss: {val_loss} \t Val perplexity: {math.exp(val_loss)}')
        val_metrics = {'loss': val_loss, 'perplexity': math.exp(val_loss)}
        viz.log_metrics(val_metrics, "val", global_step)


        if val_loss < stored_loss:
            model.save_pretrained(args.output_dir)
            print('Saving model (new best validation)')
            stored_loss = val_loss
        else:
            num_epochs_no_improvement += 1
        
        if num_epochs_no_improvement == args.wait_epochs:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * args.lr_step
            learning_rate_steps += 1

        if learning_rate_steps > 5:
            return stored_loss
        
    return stored_loss

if __name__ == '__main__':
    #removed all model architecture specific CLI args. This is supported via a JSON config file.
    parser = argparse.ArgumentParser(description='SHA-RNN language modeling')
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
    parser.add_argument('--wait_epochs', type = int, default = 3,
                        help='Reduce learning rates after wait_epochs epochs without improvement')
    parser.add_argument('--lr_step', type = float, default = 0.9,
                        help = 'factor by which to multiply learning rate at each reduction step')
    parser.add_argument('--reset_hidden', type=bool, default=False,
                        help = 'Reset the hidden state after encounter of the tokenizer stop token')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--output_dir', type=str,  default=f'{time.strftime("%Y-%m-%d",time.gmtime())}_awd_lstm_lm_pretraining',
                        help='path to save logs and trained model')
    parser.add_argument('--wandb-sweep', type=bool, default=False,
                        help='wandb hyperparameter sweep: Override hyperparams with params from wandb')
    parser.add_argument('--resume', type=str,  default='',
                        help='path of model to resume')

    #args for model architecture
    parser.add_argument('--num_hidden_layers', type=int, default=3, metavar='N',
                        help='report interval')
    parser.add_argument('--num_max_positions', type=int, default=2000, metavar='N',
                        help='max seq len positions')
    parser.add_argument('--input_size', type=int, default=400, metavar='N',
                        help='Embedding layer size')
    parser.add_argument('--hidden_size', type=int, default=1150, metavar='N',
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
    #c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%y/%m/%d %H:%M:%S")
    #c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    #logger.addHandler(c_handler)
    logger.addHandler(f_handler)

        #choose device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Running on: {device}')

    main_training_loop(args)