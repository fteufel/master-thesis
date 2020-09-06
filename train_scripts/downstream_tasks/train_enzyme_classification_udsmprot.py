'''
Train a MLP classification head on top of an AWD-LSTM LM.
Based on UDSMProt approach:
 - Gradual unfreezing
 - Different learning rates for each lstm layer
 - one-cycle cosine learning rate schedule
Hyperparameters for this were taken from UDSMProt. Adapted to use standard pytorch
rather than fastai.

Hyperparameters to be optimized:
 - learning rate
 - classifier hidden size
 - batch size

WIP Notes
- instantiating a new scheduler() on an optimizer resets its lrs and momentums. So can just
  call a new scheduler after each training cycle, no need for reinitalization of optimizer.
'''
#Felix September 2020
import argparse
import time
import numpy as np
import torch
import logging
import torch.nn as nn
import sys
sys.path.append("..")
from typing import Tuple, List, Dict
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

def even_mults(start, stop, n):
    '''Helper fuction to get a learning rate for each parameter group.
    Build log-stepped array from `start` to `stop` in `n` steps.
    from fastai. Seems to replace the rule of the original paper where
    ηl-1 = ηl / 2.6, where ηl is the learning rate of the l-th layer.
    NOTE i practically override this with the **4 transformation of the min_lr (?)
    leave for now for to stay as true to original as possible
    '''
    if n==1: return stop
    mult = stop/start
    step = mult**(1/(n-1))
    return np.array([start*(step**i) for i in range(n)])

def get_schedulers(optimizers, min_lr, max_lr, epochs, steps_per_epoch):
    '''Helper function to make a scheduler for each optimizer in a list'''
    max_lrs = even_mults(min_lr, max_lr, len(optimizers))
    schedulers = []
    for i, optimizer in enumerate(optimizers):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=max_lrs[i], 
                                                        epochs = epochs, 
                                                        steps_per_epoch=steps_per_epoch, 
                                                        pct_start = 0.25, 
                                                        base_momentum = 0.7, 
                                                        max_momentum = 0.8, 
                                                        final_div_factor= 1e5)
        schedulers.append(scheduler)

    return schedulers

def process_batch_truncated(model, data, targets, input_mask, max_tokens = 1024):
    '''Handle truncated backpropagation for classification.
    Not yet clear how to really implement - detach hidden states? Then no gradient in pooling.'''
    data_list = torch.split(data, max_tokens, dim = -1)
    mask_list = torch.split(input_mask, max_tokens, dim = -1)
    hidden_state = None

    outputs_list = []
    for idx, (inputs, mask) in enumerate(zip(data_list, mask_list)):
        logger.info(f'processing truncated piece with shape {inputs.shape}')
        #def truncated_forward(self, input_ids_truncated, input_mask_truncated = None,  targets =None, hidden_state = None,
        #                  hidden_state_seq = None, full_input_mask =None):
        if idx == len(data_list) -1:
            all_outputs = torch.cat(outputs_list , dim =1)
            loss, probs = model.truncated_forward(input_ids_truncated =inputs, targets= targets, input_mask_truncated = mask, hidden_state = hidden_state,
             hidden_state_seq = all_outputs, full_input_mask =input_mask)
        else:
            outputs, hidden_state = model.truncated_forward(input_ids_truncated =inputs, input_mask_truncated = mask, hidden_state = hidden_state)
            outputs_list.append(outputs)

    
    return loss, probs



def train(model: torch.nn.Module, train_data: DataLoader, optimizers: List[torch.optim.Optimizer], schedulers: List[torch.optim.lr_scheduler._LRScheduler], 
            args: argparse.ArgumentParser, visualizer, global_step: int, restrict_to_optimizers = None) -> Tuple[float, int]:
    model.train()
    for optimizer in optimizers: 
        optimizer.zero_grad()
    total_loss = 0
    accumulation_steps = 0
    
    for idx, batch in enumerate(train_data):
        data, targets, input_mask = batch
        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)

        logger.info(f'Inputs shape: {data.shape}')
        #loss, probs = process_batch_truncated(model, data, targets, input_mask)
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
            if restrict_to_optimizers is None:
                restrict_to_optimizers = list(range(len(optimizers))) #use all optimizers if not restricted to any

            for i in restrict_to_optimizers:
                optimizers[i].step()
                schedulers[i].step()
                optimizer.zero_grad()
                visualizer.log_metrics({f'Learning Rate {i}': optimizer.param_groups[i]['lr'] }, "train", global_step)

            
            #zero all gradients of the model to be safe. Don't want accumulation on weights that are not updated in this cycle.
            model.zero_grad()

            accumulation_steps = 0

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

def save_model_with_amp(model, output_dir):

    model.save_pretrained(output_dir)
    #also save with apex
    if torch.cuda.is_available():
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'amp': amp.state_dict()
            }
        torch.save(checkpoint, os.path.join(output_dir, 'amp_checkpoint.pt'))

def main_training_loop(args: argparse.ArgumentParser):

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

    #NOTE gradual unfreezing with AMP:
    # amp.initialize() should not be called more than once. So to keep layers frozen, we just create one optimizer per layer. 
    # We then only call the required optimizer.step() methods in each cycle.
    lstm_layers = model.encoder.encoder.lstm
    embedding_layer = model.encoder.embedding_layer
    classifier_layer = model.classifier

    # make the parameter groups
    param_groups = []
    for i, lstm_layer in enumerate(lstm_layers):
        if i == 0:
            param_groups.append(list(lstm_layer.parameters())+list(embedding_layer.parameters()) )
        elif i == len(lstm_layers) -1:
            param_groups.append(list(lstm_layer.parameters())+list(classifier_layer.parameters()))
        else:
            param_groups.append(lstm_layer.parameters())


    if args.optimizer == 'sgd':
        optimizers = [ torch.optim.SGD(params, args.lr, weight_decay=args.wdecay) for params in param_groups]
    if args.optimizer == 'adam':
        optimizers = [ torch.optim.Adam(params, weight_decay=args.wdecay) for params in param_groups]
    if args.optimizer == 'adamw':
        optimizers = [ torch.optim.AdamW(params, weight_decay=args.wdecay) for params in param_groups]


    model.to(device)
    logger.info('Model set up!')
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model has {num_parameters} trainable parameters')

    if torch.cuda.is_available():
        model, optimizer = amp.initialize(model, optimizers, opt_level='O0') #NOTE udsmprot used fp32.
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

    # Train top layer
    lr_low = args.lr/(2**4) #does not really matter when i only train one layer. but for compatibility.
    schedulers = get_schedulers(optimizers, lr_low, args.lr, epochs =1 , steps_per_epoch = len(train_loader))

    loss, global_step = train(model, train_loader, optimizers, schedulers, args, viz, global_step, restrict_to_optimizers = [-1])
    logger.info(f'Step {global_step}, trained top 1 layer: validating for {len(val_loader)} Validation steps')
    val_loss, val_metrics, roc_curve = validate(model, val_loader)
    viz.log_metrics(val_metrics, "val", global_step)
    save_model_with_amp(model, os.path.join(args.output_dir, 'top_1_layers_finetuned'))


    # Train top 2 layers
    lr_high = args.lr#/lr_stage_factor[0]
    lr_low = lr1_high/(2**4)
    schedulers = get_schedulers(optimizers, lr_low, lr_high, epochs =1 , steps_per_epoch = len(train_loader))
    loss, global_step = train(model, train_loader, optimizers, schedulers, args, viz, global_step, restrict_to_optimizers = [-1, -2])
    logger.info(f'Step {global_step}, trained top 2 layers: validating for {len(val_loader)} Validation steps')
    val_loss, val_metrics, roc_curve = validate(model, val_loader)
    viz.log_metrics(val_metrics, "val", global_step)
    save_model_with_amp(model, os.path.join(args.output_dir, 'top_2_layers_finetuned'))

    # Train top 3 layers
    lr_high = lr_high#/lr_stage_factor[0]
    lr_low = lr_high/(2**4)
    schedulers = get_schedulers(optimizers, args.min_lr, args.max_lr, epochs =1 , steps_per_epoch = len(train_loader))
    loss, global_step = train(model, train_loader, optimizers, schedulers, args, viz, global_step, restrict_to_optimizers = None)
    logger.info(f'Step {global_step}, trained top 3 layers: validating for {len(val_loader)} Validation steps')
    val_loss, val_metrics, roc_curve = validate(model, val_loader)
    viz.log_metrics(val_metrics, "val", global_step)
    save_model_with_amp(model, os.path.join(args.output_dir, 'top_3_layers_finetuned'))

    #finetune (?)
    #lr3_high = lr2_high if kwargs["lr_fixed"] else lr2_high/kwargs["lr_stage_factor"][2] #lr2_high/5
    #lr3_low = lr3_high/(kwargs["lr_slice_exponent"]**(len(learn.layer_groups)


    logger.info(f'Epoch {epoch} training complete')




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Enzyme classification')
    parser.add_argument('--data', type=str, default='/work3/felteu/data/ec_prediction/splits',
                        help='location of the data corpus. Expects [test train valid]_full.tsv')

    #args relating to training strategy.
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument('--wdecay', type=float, default=1e-7,
                        help='weight decay applied to all weights')
    parser.add_argument('--clip', type=float, default = 0.25,
                        help ='clip gradients')
    parser.add_argument('--optimizer', type=str,  default='sgd',
                        help='optimizer to use (sgd, adam, adamw)')
    parser.add_argument('--output_dir', type=str,  default=f'{time.strftime("%Y-%m-%d",time.gmtime())}_awd_lstm_lm_pretraining',
                        help='path to save logs and trained model')
    parser.add_argument('--resume', type=str,  default='/zhome/1d/8/153438/experiments//results/best_models_31072020/best_euk_model',
                        help='path of model to resume (directory containing .bin and config.json')
    parser.add_argument('--experiment_name', type=str,  default='AWD_LSTM_LM',
                        help='experiment name for logging')
    parser.add_argument('--gradient_accumulation_steps', type=bool, default=2,
                        help = 'Training minibatches over which to accumulate gradients before optimizer.step()')


    #args for model architecture
    parser.add_argument('--classifier_hidden_size', type=int, default=50, metavar='N',
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


   
    f_handler = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%y/%m/%d %H:%M:%S")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    #choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Running on: {device}')
    logger.info(f'Saving to {args.output_dir}')
    main_training_loop(args)
