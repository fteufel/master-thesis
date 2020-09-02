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
from models.awd_lstm import ProteinAWDLSTMForLM, ProteinAWDLSTMConfig
from tape import TAPETokenizer
from tape import visualization #import utils.visualization as visualization
from train_scripts.training_utils import repackage_hidden, save_training_status
from train_scripts.training_utils import VirtualBatchTruncatedBPTTHdf5Dataset as Hdf5Dataset
from torch.utils.data import DataLoader
if torch.cuda.is_available():
    from apex import amp

import os 


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        loss, raw_loss, output, hidden = model(data, targets= targets)
    else:
        hidden = repackage_hidden(previous_hidden) #detaches hidden state from graph of previous step
        loss, raw_loss, output, hidden = model(data, hidden_state = hidden, targets = targets)

    if torch.cuda.is_available():
        with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
    else:
        loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    if args.clip: 
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()
    optimizer.param_groups[0]['lr'] = lr2

    return raw_loss.item(), loss.item(), hidden

def validation_step(model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, previous_hidden: tuple = None) -> (float, tuple):
    '''Run one validation step.
    '''
    model.eval()
    data = data.to(device)
    targets = targets.to(device)

    if previous_hidden == None:
        loss, raw_loss, output, hidden = model(data, targets= targets)
    else:
        hidden = repackage_hidden(previous_hidden)
        loss, raw_loss, output, hidden = model(data, hidden_state = hidden, targets = targets)
    
    return raw_loss.item(), loss.item(), hidden

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

    logger.info(f'Data loaded. One train epoch = {len(train_data)} steps.')
    logger.info(f'Data loaded. One valid epoch = {len(val_data)} steps.')

    train_loader = DataLoader(train_data, batch_size =1, collate_fn= train_data.collate_fn)
    val_loader = DataLoader(val_data, batch_size =1, collate_fn= train_data.collate_fn)
    #setup validation here so i can get a subsample from where i stopped each time i need it
    val_iterator = enumerate(val_loader)
    val_steps = 0
    hidden = None

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
            loss, reg_loss, hidden = training_step(model, data, targets, hidden, optimizer, args, i)
            viz.log_metrics({'loss': loss, 'regularized loss': reg_loss, 'perplexity': math.exp(loss), 'regularized perplexity': math.exp(reg_loss)}, "train", global_step)
            global_step += 1


            update_steps = args.update_lr_steps if len(train_loader)>args.update_lr_steps else len(train_loader) #ad hoc fix for smaller datasets, evaluate after full epochs
            # every update_lr_steps, evaluate performance and save model/progress in learning rate
            if global_step % update_steps == 0 and global_step > 0:
                total_loss = 0
                total_reg_loss = 0
                total_len = 0

                #NOTE Plasmodium sets are 1% the size of Eukarya sets. run 1/100 of total set at each time
                #n_val_steps = (len(val_loader)//100) if len(val_loader) > 100000 else len(val_loader) #works because plasmodium set is smaller, don't want another arg for this
                #old border was too high, cannot train homology reduced eukarya 10 percent with it
                n_val_steps = (len(val_loader)//100) if len(val_loader) > 10000 else len(val_loader) #works because plasmodium set is smaller, don't want another arg for this
                logger.info(f'Step {global_step}, validating for {n_val_steps} Validation steps')

                for j in range(n_val_steps):
                    val_steps += 1
                    #if val_steps == len(val_loader): #reset the validation data when at its end
                    #    val_iterator = enumerate(val_loader)
                    #    hidden = None
                    try:
                        _, (data, targets) = next(val_iterator)
                    except:
                        val_iterator = enumerate(val_loader)
                        logger.info(f'validation step{j}: resetting validation enumerator.')
                        hidden = None
                        _, (data, targets) = next(val_iterator)
                    loss, reg_loss, hidden = validation_step(model, data, targets, hidden)
                    total_len += len(data)
                    total_reg_loss += reg_loss *len(data)
                    total_loss += loss*len(data)

                val_reg_loss = total_reg_loss/total_len
                
                val_loss = total_loss/total_len

                val_metrics = {'loss': val_loss, 'perplexity': math.exp(val_loss), 'regularized loss': val_reg_loss, 'regularized perplexity': math.exp(val_reg_loss)}
                viz.log_metrics(val_metrics, "val", global_step)

                elapsed = time.time() - start_time
                logger.info(f'Training step {global_step}, { elapsed / args.log_interval:.3f} s/batch. tr_loss: {loss:.2f}, tr_perplexity {math.exp(loss):.2f} va_loss: {val_loss:.2f}, va_perplexity {math.exp(val_loss):.2f}')
                start_time = time.time()

                if val_loss < stored_loss:
                    num_epochs_no_improvement = 0
                    model.save_pretrained(args.output_dir)
                    save_training_status(args.output_dir, epoch, global_step, num_epochs_no_improvement, stored_loss, learning_rate_steps)
                    #also save with apex
                    if torch.cuda.is_available():
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'amp': amp.state_dict()
                            }
                        torch.save(checkpoint, os.path.join(args.output_dir, 'amp_checkpoint.pt'))
                        logger.info(f'New best model with loss {val_loss}, Saving model, training step {global_step}')
                    stored_loss = val_loss
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
                            return stored_loss


            if  args.enforce_walltime == True and (time.time() - loop_start_time) > 84600: #23.5 hours
                logger.info('Wall time limit reached, ending training early')
                return stored_loss



        logger.info(f'Epoch {epoch} training complete')
        logger.info(f'Epoch {epoch}, took {time.time() - epoch_start_time:.2f}.\t Train loss: {loss:.2f} \t Train perplexity: {math.exp(loss):.2f}')

    return stored_loss



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
                        help = 'After how many update steps to check for learning rate update - do a validation pass')
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
    parser.add_argument('--bidirectional', action='store_true',
                        help='make the LM bidirectional')

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