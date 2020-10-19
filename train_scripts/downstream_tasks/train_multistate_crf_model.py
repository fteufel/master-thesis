'''
Script is to a large part identical to train_any_crf_sp_model.py
Changes: - no global labels fed to model
         - use LargeCRFPartitionDataset
         -

Train a CRF sequence tagging and global label prediction model on top of the a pretrained LM.
Expand MODEL_DICT to incorporate more LM architectures

Hyperparameters to be optimized:
 - learning rate
 - classifier hidden size
 - batch size


'''
#Felix August 2020
import argparse
import time
import math
import numpy as np
import torch
import logging
import subprocess
import torch.nn as nn
import sys
sys.path.append("..")
from typing import Tuple, Dict, List
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet
from models.multi_crf_bert import ProteinBertTokenizer, BertSequenceTaggingCRF
from transformers import BertConfig
from train_scripts.utils.signalp_dataset import LargeCRFPartitionDataset, SIGNALP_KINGDOM_DICT
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_scripts.downstream_tasks.metrics_utils import get_metrics, find_cs_tag

from train_scripts.utils.smart_optim import Adamax


#import data
import os 
import wandb

from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score, recall_score, precision_score


def log_metrics(metrics_dict, split: str, step: int):
    '''Convenience function to add prefix to all metrics before logging.'''
    wandb.log({f"{split.capitalize()} {name.capitalize()}": value
                for name, value in metrics_dict.items()}, step=step)

# TODO quick fix for hyperparameter search, fix and move to utils. or get rid of this class completely.
class DecoyConfig():
    def update(*args, **kwargs):
        pass

class DecoyWandb():
    config = DecoyConfig()
    def init(self, *args, **kwargs):
        pass
    def log(self, *args, **kwargs):
        print(args)
        print(kwargs)
    def watch(self, *args, **kwargs):
        pass
    

#get the git hash - and log it
#wandb does that automatically - but only when in the correct directory when launching the job.
#by also doing it manually, force to launch from the correct directory, because otherwise this command will fail.
GIT_HASH = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode()


MODEL_DICT = {
              'bert_prottrans': (BertConfig, BertSequenceTaggingCRF),
             }
TOKENIZER_DICT = {
                  'bert_prottrans': (ProteinBertTokenizer, 'Rostlab/prot_bert')
                 }

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%y/%m/%d %H:%M:%S")
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    return logger


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tagged_seq_to_cs_multiclass(tagged_seqs: np.ndarray, sp_tokens = [0,4,5]):
    '''Convert a sequences of tokens to the index of the cleavage site.
    Inputs:
        tagged_seqs: (batch_size, seq_len) integer array of position-wise labels
        sp_tokens: label tokens that indicate a signal peptide
    Returns:
        cs_sites: (batch_size) integer array of last position that is a SP. NaN if no SP present in sequence.
    '''
    def get_last_sp_idx(x: np.ndarray) -> int:
        '''Func1d to get the last index that is tagged as SP. use with np.apply_along_axis. '''
        sp_idx = np.where(np.isin(x, sp_tokens))[0]
        if len(sp_idx)>0:
            #TODO remove or rework to warning, in training might well be that continuity is not learnt yet (i don't enforce it in the crf setup)
            #assert sp_idx.max() +1 - sp_idx.min() == len(sp_idx) #assert continuity. CRF should actually guarantee that (if learned correcty)
            max_idx = sp_idx.max()+1
        else:
            max_idx = np.nan
        return max_idx

    cs_sites = np.apply_along_axis(get_last_sp_idx, 1, tagged_seqs)
    return cs_sites

   

def report_metrics(true_global_labels: np.ndarray, pred_global_labels: np.ndarray, true_sequence_labels: np.ndarray, 
                    pred_sequence_labels: np.ndarray, use_cs_tag = False) -> Dict[str, float]:
    '''Utility function to get metrics from model output'''
    true_cs = tagged_seq_to_cs_multiclass(true_sequence_labels, sp_tokens = [4, 9, 14] if use_cs_tag else [3,7,11])
    pred_cs = tagged_seq_to_cs_multiclass(pred_sequence_labels, sp_tokens = [4, 9, 14] if use_cs_tag else [3,7,11])
    #TODO decide on how to calcuate cs metrics: ignore no cs seqs, or non-detection implies correct cs?
    pred_cs = pred_cs[~np.isnan(true_cs)]
    true_cs = true_cs[~np.isnan(true_cs)]
    true_cs[np.isnan(true_cs)] = -1
    pred_cs[np.isnan(pred_cs)] = -1

    #applying a threhold of 0.25 (SignalP) to a 4 class case is equivalent to the argmax.
    pred_global_labels_thresholded = pred_global_labels.argmax(axis=1)
    metrics_dict = {}
    metrics_dict['CS Recall'] = recall_score(true_cs, pred_cs, average='micro')
    metrics_dict['CS Precision'] = precision_score(true_cs, pred_cs, average='micro')
    metrics_dict['CS MCC'] = matthews_corrcoef(true_cs, pred_cs)
    metrics_dict['Detection MCC'] = matthews_corrcoef(true_global_labels, pred_global_labels_thresholded)

    return metrics_dict

def report_metrics_kingdom_averaged(true_global_labels: np.ndarray, pred_global_labels: np.ndarray, true_sequence_labels: np.ndarray, 
                    pred_sequence_labels: np.ndarray, kingdom_ids: np.ndarray, use_cs_tag = False) -> Dict[str, float]:
    '''Utility function to get metrics from model output'''
    true_cs = tagged_seq_to_cs_multiclass(true_sequence_labels, sp_tokens = [4, 9, 14] if use_cs_tag else [3,7,11])
    pred_cs = tagged_seq_to_cs_multiclass(pred_sequence_labels, sp_tokens = [4, 9, 14] if use_cs_tag else [3,7,11])

    cs_kingdom = kingdom_ids[~np.isnan(true_cs)]
    pred_cs = pred_cs[~np.isnan(true_cs)]
    true_cs = true_cs[~np.isnan(true_cs)]
    true_cs[np.isnan(true_cs)] = -1
    pred_cs[np.isnan(pred_cs)] = -1

    #applying a threhold of 0.25 (SignalP) to a 4 class case is equivalent to the argmax.
    pred_global_labels_thresholded = pred_global_labels.argmax(axis=1)

    #compute metrics for each kingdom
    rev_kingdom_dict = dict(zip(SIGNALP_KINGDOM_DICT.values(), SIGNALP_KINGDOM_DICT.keys()))
    all_cs_mcc = []
    all_detection_mcc = []
    metrics_dict = {}
    for kingdom in np.unique(kingdom_ids):
        kingdom_global_labels = true_global_labels[kingdom_ids==kingdom] 
        kingdom_pred_global_labels_thresholded = pred_global_labels_thresholded[kingdom_ids==kingdom]
        kingdom_true_cs = true_cs[cs_kingdom==kingdom]
        kingdom_pred_cs = pred_cs[cs_kingdom==kingdom]

        metrics_dict[f'CS Recall {rev_kingdom_dict[kingdom]}'] = recall_score(kingdom_true_cs, kingdom_pred_cs, average='micro')
        metrics_dict[f'CS Precision {rev_kingdom_dict[kingdom]}'] = precision_score(kingdom_true_cs, kingdom_pred_cs, average='micro')
        metrics_dict[f'CS MCC {rev_kingdom_dict[kingdom]}'] = matthews_corrcoef(kingdom_true_cs, kingdom_pred_cs)
        metrics_dict[f'Detection MCC {rev_kingdom_dict[kingdom]}'] = matthews_corrcoef(kingdom_global_labels, kingdom_pred_global_labels_thresholded)

        all_cs_mcc.append(metrics_dict[f'CS MCC {rev_kingdom_dict[kingdom]}'])
        all_detection_mcc.append(metrics_dict[f'Detection MCC {rev_kingdom_dict[kingdom]}'])
        
    metrics_dict['CS MCC'] = sum(all_cs_mcc)/len(all_cs_mcc)
    metrics_dict['Detection MCC'] = sum(all_detection_mcc)/len(all_detection_mcc)

    return metrics_dict


def train(model: torch.nn.Module, train_data: DataLoader, optimizer: torch.optim.Optimizer, 
                    args: argparse.ArgumentParser, global_step: int) -> Tuple[float, int]:
    '''Predict one minibatch and performs update step.
    Returns:
        loss: loss value of the minibatch
    '''

    model.train()
    optimizer.zero_grad()

    all_targets = []
    all_global_targets = []
    all_global_probs = []
    all_pos_preds = []
    all_kingdom_ids = [] #gather ids for kingdom-averaged metrics
    total_loss = 0
    for i, batch in enumerate(train_data):
        data, targets, input_mask, global_targets, sample_weights, kingdom_ids = batch
        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)
        sample_weights = sample_weights.to(device) if args.use_sample_weights else None
        kingdom_ids = kingdom_ids.to(device)
        loss, global_probs, pos_probs, pos_preds = model(data, 
                                                    targets=  targets,
                                                    input_mask = input_mask,
                                                    sample_weights = sample_weights,
                                                    kingdom_ids = kingdom_ids if args.kingdom_embed_size > 0 else None
                                                    )
        loss = loss.mean() #if DataParallel because loss is a vector, if not doesn't matter
        total_loss += loss.item()
        all_targets.append(targets.detach().cpu().numpy())
        all_global_targets.append(global_targets.detach().cpu().numpy())
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())
        all_kingdom_ids.append(kingdom_ids.detach().cpu().numpy())


        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        #from IPython import embed; embed()
        optimizer.step()

        log_metrics({'loss': loss}, "train", global_step)
         

        if args.optimizer == 'smart_adamax':
            log_metrics({'Learning rate': optimizer.get_lr()[0]}, "train", global_step)
        else:
            log_metrics({'Learning Rate': optimizer.param_groups[0]['lr'] }, "train", global_step)
        global_step += 1

    
    #NOTE with larger batches or parallel this might become problematic
    all_targets = np.concatenate(all_targets)
    all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)
    all_kingdom_ids = np.concatenate(all_kingdom_ids)

    if args.average_per_kingdom:
        metrics = report_metrics_kingdom_averaged(all_global_targets, all_global_probs, all_targets, all_pos_preds, all_kingdom_ids, args.use_cs_tag)
    else:
        metrics = report_metrics(all_global_targets,all_global_probs, all_targets, all_pos_preds, args.use_cs_tag)    
    log_metrics(metrics, 'train', global_step)


    return total_loss/len(train_data), global_step


def validate(model: torch.nn.Module, valid_data: DataLoader, args) -> float:
    '''Run over the validation data. Average loss over the full set.
    '''
    model.eval()

    all_targets = []
    all_global_targets = []
    all_global_probs = []
    all_pos_preds = []
    all_kingdom_ids = []

    total_loss = 0
    for i, batch in enumerate(valid_data):
        data, targets, input_mask, global_targets, sample_weights, kingdom_ids = batch
        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)
        global_targets = global_targets.to(device)
        sample_weights = sample_weights.to(device) if args.use_sample_weights else None
        kingdom_ids = kingdom_ids.to(device)

        with torch.no_grad():
            loss, global_probs, pos_probs, pos_preds = model(data, 
                                                            targets=  targets, 
                                                            #global_targets = global_targets, 
                                                            sample_weights=sample_weights,
                                                            input_mask = input_mask,
                                                            kingdom_ids = kingdom_ids if args.kingdom_embed_size > 0 else None
                                                            )

        total_loss += loss.mean().item()
        all_targets.append(targets.detach().cpu().numpy())
        all_global_targets.append(global_targets.detach().cpu().numpy())
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())
        all_kingdom_ids.append(kingdom_ids.detach().cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)
    all_kingdom_ids = np.concatenate(all_kingdom_ids)

    if args.average_per_kingdom:
        metrics = report_metrics_kingdom_averaged(all_global_targets, all_global_probs, all_targets, all_pos_preds, all_kingdom_ids, args.use_cs_tag)
    else:
        metrics = report_metrics(all_global_targets,all_global_probs, all_targets, all_pos_preds, args.use_cs_tag)    


    val_metrics = {'loss': total_loss / len(valid_data), **metrics }
    return (total_loss / len(valid_data)), val_metrics

def main_training_loop(args: argparse.ArgumentParser):


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    logger = setup_logger()
    f_handler = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%y/%m/%d %H:%M:%S")
    f_handler.setFormatter(formatter)

    logger.addHandler(f_handler)

    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    experiment_name = f"{args.experiment_name}_{args.test_partition}_{args.validation_partition}_{time_stamp}"


    #TODO get rid of this dirty fix once wandb works again
    global wandb
    import wandb
    if wandb.run is None and not args.crossval_run: #Only initialize when there is no run yet (when importing main_training_loop to other scripts)
        wandb.init(dir=args.output_dir, name=experiment_name)
    else:
        wandb=DecoyWandb()


    logger.info(f'Saving to {args.output_dir}')

    #Setup Model
    logger.info(f'Loading pretrained model in {args.resume}')
    config = MODEL_DICT[args.model_architecture][0].from_pretrained(args.resume)

    if config.xla_device:
        setattr(config, 'xla_device', False)

    #patch LM model config for new downstream task
    setattr(config, 'num_labels', 7 if args.eukarya_only else 15)
    setattr(config, 'num_global_labels', 2 if args.eukarya_only else 4)


    setattr(config, 'lm_output_dropout', args.lm_output_dropout)
    setattr(config, 'lm_output_position_dropout', args.lm_output_position_dropout)
    setattr(config, 'crf_scaling_factor', args.crf_scaling_factor)
    setattr(config, 'use_large_crf', True) #legacy, parameter is used in evaluation scripts. Ensures choice of right CS states.

    if args.kingdom_embed_size > 0:
        setattr(config, 'use_kingdom_id', True)
        setattr(config, 'kingdom_embed_size', args.kingdom_embed_size)
 

    if args.remove_top_layers > 0:
        #num_hidden_layers if bert
        n_layers = config.num_hidden_layers if args.model_architecture == 'bert_prottrans' else config.n_layer
        if args.remove_top_layers > n_layers:
            logger.warning(f'Trying to remove more layers than there are: {n_layers}')
            args.remove_top_layers = n_layers

        setattr(config, 'num_hidden_layers' if args.model_architecture == 'bert_prottrans' else 'n_layer',n_layers-args.remove_top_layers)

    model = MODEL_DICT[args.model_architecture][1].from_pretrained(args.resume, config = config)
    tokenizer = TOKENIZER_DICT[args.model_architecture][0].from_pretrained(TOKENIZER_DICT[args.model_architecture][1], do_lower_case =False)
    logger.info(f'Loaded weights from {args.resume} for model {model.base_model_prefix}')
    
    #setup data
    val_id = args.validation_partition
    test_id = args.test_partition
    train_ids = [0,1,2,3,4]
    train_ids.remove(val_id)
    train_ids.remove(test_id)
    logger.info(f'Training on {train_ids}, validating on {val_id}')
    if args.archaea_only:
        kingdoms = ['ARCHAEA']
        assert args.archaea_only != args.eukarya_only, "archaea_only and eukarya_only cannot be true at the same time."
    elif args.eukarya_only:
        kingdoms = ['EUKARYA']
        assert args.archaea_only != args.eukarya_only, "archaea_only and eukarya_only cannot be true at the same time."
    else:
        kingdoms = ['EUKARYA', 'ARCHAEA', 'NEGATIVE', 'POSITIVE']

    special_tokens_flag = True #add cls and sep
    train_data = LargeCRFPartitionDataset(args.data, args.sample_weights ,tokenizer= tokenizer, partition_id = train_ids, kingdom_id=kingdoms, 
                                          add_special_tokens = special_tokens_flag,return_kingdom_ids=True, positive_samples_weight= args.positive_samples_weight,
                                          make_cs_state = args.use_cs_tag)
    val_data = LargeCRFPartitionDataset(args.data, args.sample_weights , tokenizer = tokenizer, partition_id = [val_id], kingdom_id=kingdoms, 
                                        add_special_tokens = special_tokens_flag, return_kingdom_ids=True, positive_samples_weight= args.positive_samples_weight,
                                        make_cs_state = args.use_cs_tag)
    logger.info(f'{len(train_data)} training sequences, {len(val_data)} validation sequences.')

    train_loader = DataLoader(train_data, batch_size =args.batch_size, collate_fn= train_data.collate_fn, shuffle = True)
    val_loader = DataLoader(val_data, batch_size =args.batch_size, collate_fn= train_data.collate_fn)

    if args.use_random_weighted_sampling:
        sampler = WeightedRandomSampler(train_data.sample_weights, len(train_data), replacement=False)
        train_loader = DataLoader(train_data, batch_size = args.batch_size, collate_fn = train_data.collate_fn, sampler = sampler)
    
    logger.info(f'Data loaded. One epoch = {len(train_loader)} batches.')

    #set up wandb logging, login and project id from commandline vars
    wandb.config.update(args)
    wandb.config.update({'git commit ID': GIT_HASH})
    wandb.config.update(model.config.to_dict())
    # TODO uncomment as soon as w&b fixes the bug on their end.
    # wandb.watch(model)
    logger.info(f'Logging experiment as {experiment_name} to wandb/tensorboard')
    logger.info(f'Saving checkpoints at {args.output_dir}')


    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'smart_adamax':
        t_total = len(train_loader) * args.epochs
        optimizer = Adamax(model.parameters(), lr = args.lr, warmup = 0.1,t_total = t_total,schedule='warmup_linear', betas = (0.9, 0.999),max_grad_norm=1)



    model.to(device)
    logger.info('Model set up!')
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model has {num_parameters} trainable parameters')

    logger.info(f'Running model on {device}, not using nvidia apex')


        
    #keep track of best loss
    stored_loss = 100000000
    learning_rate_steps = 0
    num_epochs_no_improvement = 0
    global_step = 0
    best_mcc_sum = 0
    best_mcc_global = 0
    best_mcc_cs = 0
    for epoch in range(1, args.epochs+1):
        logger.info(f'Starting epoch {epoch}')

        
        epoch_loss, global_step = train(model, train_loader,optimizer,args,global_step)

        logger.info(f'Step {global_step}, Epoch {epoch}: validating for {len(val_loader)} Validation steps')
        val_loss, val_metrics = validate(model, val_loader, args)
        log_metrics(val_metrics, "val", global_step)
        logger.info(f"Validation: MCC global {val_metrics['Detection MCC']}, MCC seq {val_metrics['CS MCC']}. Epochs without improvement: {num_epochs_no_improvement}. lr step {learning_rate_steps}")

        mcc_sum = val_metrics['Detection MCC'] + val_metrics['CS MCC']
        log_metrics({'MCC Sum': mcc_sum}, 'val', global_step)
        if mcc_sum > best_mcc_sum:
            best_mcc_sum = mcc_sum
            best_mcc_global = val_metrics['Detection MCC']
            best_mcc_cs = val_metrics['CS MCC']
            num_epochs_no_improvement = 0

            model.save_pretrained(args.output_dir)
            logger.info(f'New best model with loss {val_loss},MCC Sum {mcc_sum} MCC global {val_metrics["Detection MCC"]}, MCC seq {val_metrics["CS MCC"]}, Saving model, training step {global_step}')

        else:
            num_epochs_no_improvement += 1


        if (num_epochs_no_improvement > args.annealing_epochs) and learning_rate_steps == 0:
            logger.info('No improvement for 10 epochs, reducing learning rate to 1/10.')
            num_epochs_no_improvement = 0
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
            learning_rate_steps = 1


    logger.info(f'Epoch {epoch}, epoch limit reached. Training complete')
    logger.info(f'Best: MCC Sum {best_mcc_sum}, Detection {best_mcc_global}, CS {best_mcc_cs}')
    log_metrics({'Best MCC Detection': best_mcc_global, 'Best MCC CS': best_mcc_cs, 'Best MCC sum': best_mcc_sum}, "val", global_step)

    print_all_final_metrics = True
    if print_all_final_metrics == True:
        #reload best checkpoint
        #TODO Kingdom ID handling needs to be added here.
        model = MODEL_DICT[args.model_architecture][1].from_pretrained(args.output_dir)
        ds = LargeCRFPartitionDataset(args.data, tokenizer= tokenizer, partition_id = [test_id], 
                                      kingdom_id=kingdoms, add_special_tokens = special_tokens_flag, return_kingdom_ids=True,
                                      make_cs_state=args.use_cs_tag)
        dataloader = torch.utils.data.DataLoader(ds, collate_fn = ds.collate_fn, batch_size = 80)
        metrics = get_metrics(model,dataloader, cs_tagged = args.use_cs_tag)
        val_metrics = get_metrics(model,val_loader, cs_tagged = args.use_cs_tag)

        if args.crossval_run or args.log_all_final_metrics:
            log_metrics(metrics, "test", global_step)
            log_metrics(val_metrics, "best_val", global_step)
        logger.info(metrics)
        logger.info('Validation set')
        logger.info(val_metrics)

    return best_mcc_global, best_mcc_cs #best_mcc_sum



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train CRF on top of Pfam Bert')
    parser.add_argument('--data', type=str, default='../data/data/signalp_5_data/',
                        help='location of the data corpus. Expects test, train and valid .fasta')
    parser.add_argument('--sample_weights', type=str, default=None,
                        help='path to .csv file with the weights for each sample')
    parser.add_argument('--test_partition', type = int, default = 0,
                        help = 'partition that will not be used in this training run')
    parser.add_argument('--validation_partition', type = int, default = 1,
                        help = 'partition that will be used for validation in this training run')
            
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
    parser.add_argument('--min_epochs', type=int, default=100,
                        help='minimum epochs to train for before reducing lr. Useful when there is a plateau in performance.')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--optimizer', type=str,  default='sgd',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--output_dir', type=str,  default=f'{time.strftime("%Y-%m-%d",time.gmtime())}_awd_lstm_lm_pretraining',
                        help='path to save logs and trained model')
    parser.add_argument('--resume', type=str,  default='bert-base',
                        help='path of model to resume (directory containing .bin and config.json')
    parser.add_argument('--experiment_name', type=str,  default='PFAM-BERT-CRF',
                        help='experiment name for logging')
    parser.add_argument('--crossval_run', action = 'store_true',
                        help = 'override name with timestamp, save with split identifiers. Use when making checkpoints for crossvalidation.')
    parser.add_argument('--log_all_final_metrics', action='store_true', help='log all final test/val metrics to w&b')

    parser.add_argument('--eukarya_only', action='store_true', help = 'Only train on eukarya SPs')
    parser.add_argument('--archaea_only', action='store_true', help = 'Only train on archaea SPs')



    parser.add_argument('--lm_output_dropout', type=float, default = 0.1,
                        help = 'dropout applied to LM output')
    parser.add_argument('--lm_output_position_dropout', type=float, default = 0.1,
                        help='dropout applied to LM output, drops full hidden states from sequence')
    parser.add_argument('--use_sample_weights', action = 'store_true', 
                        help = 'Use sample weights to rescale loss per sample')
    parser.add_argument('--use_random_weighted_sampling', action='store_true',
                        help='use sample weights to load random samples as minibatches according to weights')
    parser.add_argument('--annealing_epochs', type=int, default=10, metavar='N',
                        help='after how many epochs without improvement to reduce learning rate by 10.')
    parser.add_argument('--multi_gpu', action='store_true', help = 'Use DataParallel (single-node multi GPU')
    parser.add_argument('--positive_samples_weight', type=float, default=None,
                        help='Scaling factor for positive samples loss, e.g. 1.5. Needs --use_sample_weights flag in addition.')
    parser.add_argument('--average_per_kingdom', action='store_true',
                        help='Average MCCs per kingdom instead of overall computatition')
    parser.add_argument('--crf_scaling_factor', type=float, default=1.0, help='Scale CRF NLL by this before adding to global label loss')

    #args for model architecture
    parser.add_argument('--model_architecture', type=str, default = 'bert',
                        help ='which model architecture the checkpoint is for')
    parser.add_argument('--use_rnn', action='store_true',
                        help='use biLSTM instead of MLP for emissions')
    parser.add_argument('--classifier_hidden_size', type=int, default=0, metavar='N',
                        help='Hidden size of the classifier head MLP')
    parser.add_argument('--remove_top_layers', type=int, default=0, 
                        help='How many layers to remove from the top of the LM.')
    parser.add_argument('--kingdom_embed_size', type=int, default=0,
                        help='If >0, embed kingdom ids to N and concatenate with LM hidden states before CRF.')
    parser.add_argument('--use_cs_tag', action='store_true', help='Replace last token of SP with C for cleavage site')


    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #make unique output dir in output dir
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    full_name = '_'.join([args.experiment_name, 'test', str(args.test_partition), 'valid', str(args.validation_partition), time_stamp])
    
    if args.crossval_run == True:
        full_name ='_'.join([args.experiment_name, 'test', str(args.test_partition), 'valid', str(args.validation_partition)])

    args.output_dir = os.path.join(args.output_dir, full_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)



    main_training_loop(args)