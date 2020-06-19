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
from tape import TAPETokenizer

import data
import os 
import hashlib


class Corpus(object):
    def __init__(self, path, tokenizer: TAPETokenizer):
        '''
        Corpus class from AWD-LSTM repo. Adapted to support TAPETokenizers
        Whole dictionary thing is actually pointless, as the alphabet is static
        Expects a directory of line-by-line .txt files. End of line gets stop token
        '''
        self.tokenizer = tokenizer
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))        

    def tokenize(self, path):
        '''
        Terrible implementation. Check LongTensor docs to make better
        '''
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = self.tokenizer.tokenize(line.rstrip()) + [self.tokenizer.stop_token]
                tokens += len(words)

        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = self.tokenizer.tokenize(line.rstrip()) + [self.tokenizer.stop_token]
                tokens += len(words)
                for word in words:
                    ids[token] = self.tokenizer.convert_token_to_id(word)
                    token += 1
        return ids

def repackage_hidden(hiddens):
    hiddens = [(h.detach(), c.detach()) for h, c in hiddens]
    return hiddens


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data.to(device), target.to(device)

def train_epoch(model: torch.nn.Module, train_data , optimizer: torch.optim.Optimizer, args: argparse.ArgumentParser):
    '''
    better training function, things are passed explicitly here
    Trains for one epoch
    '''

    total_loss = 0
    start_time = time.time()
    batch, i = 0, 0
    cur_loss = None
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        #scale learning rate
        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        print(f'seq pos:{i} feeding new truncated sequence')
        if i == 0:
            loss, output, hidden = model(data, targets= targets)
        else:
            hidden = repackage_hidden(hidden)
            loss, output, hidden = model(data, hidden_state = hidden, targets = targets)

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data
        #reset learning rate
        optimizer.param_groups[0]['lr'] = lr2

        logger.info(f'Processed {batch}, {i}/{train_data.size(0)} sequence tokens.')
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                0, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
    #TODO nasty error when log_interval is smaller than total number of batches, cur_loss is never calculated
    # if len(train_data) // args.bptt
    # logger.warning('logging interval smaller than number of minibatches in epoch, no intra-epoch logs will be produced)
    #curr_loss = None
    #if not curr_loss:
    #   curr_loss = total_loss.item/batch
    print('epoch complete')
    return cur_loss #arbitrary choice


def validate(model: torch.nn.Module, valid_data , optimizer: torch.optim.Optimizer, args: argparse.ArgumentParser):
    '''
    Currently kind of one-batch setup. Fix later, when I have actual data
    '''
    model.eval()

    total_loss = 0

    for i in range(0, valid_data.size(0) - 1, args.bptt):
        data, targets = get_batch(valid_data, i, args, evaluation=True)

        if i == 0:
            loss, output, hidden = model(data, targets= targets)
        else:
            loss, output, hidden = model(data, hidden_state = hidden, targets = targets)

        #TODO detach from graph, maybe .data does the trick (deprecated who cares)
        total_loss += len(data) * loss.data #scale by length
        hidden = repackage_hidden(hidden) #detach from graph

    return total_loss.item() / len(valid_data)


def main_training_loop(args: argparse.ArgumentParser):
    '''
    Proper training loop
    '''

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #choose device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Running on: {device}')

    #load data


    tokenizer = TAPETokenizer(vocab = 'iupac')
    fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest()) #cached data format
    if os.path.exists(fn):
        logger.info('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        logger.info('Producing dataset...')
        corpus = Corpus(args.data, tokenizer)
        torch.save(corpus, fn)

    #TODO don't see why I need to differentiate here
    eval_batch_size = args.batch_size
    test_batch_size = 1
    train_data = batchify(corpus.train, args.batch_size, args)
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)

    #setup model
    config = ProteinAWDLSTMConfig() #from_file, add to args
    config.input_size = tokenizer.vocab_size
    if args.reset_hidden:
        logger.info(f'Resetting hidden state after {tokenizer.stop_token}')
        config.reset_token_id = tokenizer.convert_token_to_id(tokenizer.stop_token)

    model = ProteinAWDLSTMForLM(config)

    if args.resume:
        logger.info(f'Loading pretrained model in {args.resume}')
        model = ProteinAWDLSTMForLM.from_pretrained(args.resume)
    

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    
    logger.info('Model set up!')
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model has {num_parameters} trainable parameters')

    #keep track of best loss
    best_val_loss = None
    num_epochs_no_improvement = 0
    stored_loss = 100000000
    learning_rate_steps = 0


    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        #train
        #train_loss = train_epoch(model, train_data, optimizer, args)
        #eval
        val_loss = validate(model, val_data, optimizer, args)

        logger.info(f'Epoch{epoch}, took {time.time() - epoch_start_time}.\n Val loss: {val_loss} \t Val perplexity: {math.exp(val_loss)}')
        #print('-' * 89)
        #print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #    'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
        #    epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
        #print('-' * 89)

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
    parser.add_argument('--reset_hidden', type=bool, default=True,
                        help = 'Reset the hidden state after encounter of the tokenizer stop token')

    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--output_dir', type=str,  default=f'{time.strftime("%Y-%m-%d",time.gmtime())}_awd_lstm_lm_pretraining',
                        help='path to save logs and trained model')


    parser.add_argument('--resume', type=str,  default='',
                        help='path of model to resume (directory containing .bin and config.json')

    args = parser.parse_args()
    args.tied = True


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


    main_training_loop(args)