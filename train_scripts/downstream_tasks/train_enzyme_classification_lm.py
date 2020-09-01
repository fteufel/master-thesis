'''
Train a MLP classification head on top of an LM (output = mean pooled hidden states)
to predict the top level EC code (or 0 if not an enzyme)

Hyperparameters to be optimized:
 - learning rate
 - classifier hidden size
 - batch size
'''
#Felix August 2020
import argparse
import time
import numpy as np
import torch
import logging
import torch.nn as nn
import sys
sys.path.append("..")
from typing import Tuple
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet
from models.awd_lstm import ProteinAWDLSTMConfig, ProteinAWDLSTMForSequenceClassification
from tape import visualization #import utils.visualization as visualization
from train_scripts.training_utils import SequenceClassificationDataset
from torch.utils.data import DataLoader
from apex import amp

import os 
import wandb

from sklearn.metrics import roc_auc_score

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


def train(model: torch.nn.Module, train_data: DataLoader, optimizer: torch.optim.Optimizer, args: argparse.ArgumentParser, 
            visualizer, global_step: int) -> Tuple[float, int]:
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    accumulation_steps = 0
    
    for idx, batch in enumerate(train_data):
        data, targets, input_mask = batch
        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)

        loss, probs = model(data, targets=targets, input_mask=input_mask)
        total_loss += loss.item()

        loss = loss/args.gradient_accumulation_steps
        accumulation_steps += 1

        if torch.cuda.is_available():
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.backward()

        if args.clip: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        if accumulation_steps == args.gradient_accumulation_steps:
            optimizer.step()
            accumulation_steps = 0
            optimizer.zero_grad()
        
        visualizer.log_metrics({'loss': loss}, "train", global_step)
        global_step += 1

    return total_loss/len(train_data), global_step

    
def validate(model: torch.nn.Module, valid_data: DataLoader) -> float:
    '''Run over the validation data. Average loss over the full set.
    '''
    model.eval()

    all_targets = []
    all_probs = []

    total_loss = 0
    for i, batch in enumerate(valid_data):
        data, targets, input_mask = batch
        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)

        loss, probs = model(data, targets=targets, input_mask=input_mask)

        total_loss += loss.item()


        all_targets.append(targets.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())


    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    try:
        global_roc_curve = wandb.plots.ROC(all_targets, all_probs, [0,1,2,3,4,5,6,7])
    except: 
        global_roc_curve = np.nan #sometimes wandb roc fails for numeric reasons

    auc_macro = roc_auc_score(all_targets, all_probs, average='macro', multi_class = 'ovo')
    auc_weighted = roc_auc_score(all_targets, all_probs, average='weighted', multi_class = 'ovo')


    val_metrics = {'loss': total_loss / len(valid_data), 'AUC macro': auc_macro, 'AUC weighted': auc_weighted}
    return (total_loss / len(valid_data)), val_metrics, global_roc_curve


def main_training_loop(args: argparse.ArgumentParser):
    if args.enforce_walltime == True:
        loop_start_time = time.time()
        logger.info('Started timing loop')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #Setup Model
    assert args.resume is not None, 'Need to load a pretrained LM!'

    logger.info(f'Loading pretrained model in {args.resume}')
    config = ProteinAWDLSTMConfig.from_pretrained(args.resume)
    #override old model config from commandline args
    setattr(config, 'num_labels', args.num_labels)
    setattr(config, 'classifier_hidden_size', args.classifier_hidden_size)
    setattr(config, 'batch_first', True)

    model = ProteinAWDLSTMForSequenceClassification.from_pretrained(args.resume, config = config)    
    #training logger
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    experiment_name = f"{args.experiment_name}_{time_stamp}"
    viz = visualization.get(args.output_dir, experiment_name, local_rank = -1) #debug=args.debug) #this -1 means traning is not distributed, debug makes experiment dry run for wandb

    train_data = SequenceClassificationDataset(os.path.join(args.data, 'train_full.tsv'))
    val_data = SequenceClassificationDataset(os.path.join(args.data, 'valid_full.tsv'))


    train_loader = DataLoader(train_data, batch_size =args.batch_size, collate_fn= train_data.collate_fn)
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
        
    #keep track of best checkpoint
    best_roc_curve = None
    best_auc = 0
    learning_rate_steps = 0
    num_epochs_no_improvement = 0
    global_step = 0

    for epoch in range(1, args.epochs+1):
        logger.info(f'Starting epoch {epoch}')
        viz.log_metrics({'Learning Rate': optimizer.param_groups[0]['lr'] }, "train", global_step)

        epoch_start_time = time.time()
        start_time = time.time() #for lr update interval

        loss, global_step = train(model, train_loader, optimizer, args, viz, global_step)

        logger.info(f'Step {global_step}, Epoch {epoch}: validating for {len(val_loader)} Validation steps')
        val_loss, val_metrics, roc_curve = validate(model, val_loader)
        viz.log_metrics(val_metrics, "val", global_step)

        if epoch == args.epochs:
           viz.log_metrics({'Detection roc curve': best_roc_curve}, 'val', global_step)


        if val_metrics['AUC macro']>best_auc:
            best_auc = val_metrics['AUC macro']
            best_roc_curve = roc_curve


            model.save_pretrained(args.output_dir)
            #also save with apex
            if torch.cuda.is_available():
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict()
                    }
                torch.save(checkpoint, os.path.join(args.output_dir, 'amp_checkpoint.pt'))
                logger.info(f'New best model with loss {val_loss}, AUC {best_AUC}, Saving model, training step {global_step}')

        if (epoch>100) and  (num_epochs_no_improvement > 10):
            logger.info('No improvement for 10 epochs, ending training early.')
            logger.info(f'Best: AUC {best_AUC}')
            viz.log_metrics({'Detection roc curve': best_roc_curve}, 'val', global_step)
            return (val_loss, best_AUC)            

        if  args.enforce_walltime == True and (time.time() - loop_start_time) > 84600: #23.5 hours
            logger.info('Wall time limit reached, ending training early')
            logger.info(f'Best: AUC {best_AUC}')
            viz.log_metrics({'Detection roc curve': best_roc_curve}, 'val', global_step)
            return (val_loss, best_AUC)    

        logger.info(f'Epoch {epoch} training complete')
        logger.info(f'Epoch {epoch}, took {time.time() - epoch_start_time:.2f}.\t Train loss: {loss:.2f}')
    
    viz.log_metrics({'Detection roc curve': best_roc_curve}, 'val', global_step)
    return (val_loss, best_AUC)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Enzyme classification')
    parser.add_argument('--data', type=str, default='/work3/felteu/data/ec_prediction/splits',
                        help='location of the data corpus. Expects [test train valid]_full.tsv')

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
    parser.add_argument('--resume', type=str,  default='/zhome/1d/8/153438/experiments//results/best_models_31072020/best_euk_model',
                        help='path of model to resume (directory containing .bin and config.json')
    parser.add_argument('--experiment_name', type=str,  default='AWD_LSTM_LM',
                        help='experiment name for logging')
    parser.add_argument('--enforce_walltime', type=bool, default =True,
                        help='Report back current result before 24h wall time is over')
    parser.add_argument('--gradient_accumulation_steps', type=bool, default=2,
                        help = 'Training minibatches over which to accumulate gradients before optimizer.step()')


    #args for model architecture
    parser.add_argument('--classifier_hidden_size', type=int, default=512, metavar='N',
                        help='Hidden size of the classifier head MLP')
    parser.add_argument('--num_labels', type=int, default=8, metavar='N',
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