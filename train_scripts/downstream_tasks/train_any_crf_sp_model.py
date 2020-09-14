'''
Train a CRF sequence tagging and global label prediction model on top of the a pretrained LM.
Expand MODEL_DICT to incorporate more LM architectures

Hyperparameters to be optimized:
 - learning rate
 - classifier hidden size
 - batch size

TODO decide on correct checkpoint selection: Right now, saves lowest validation loss checkpoint.
If I want to trade off the different metrics, cannot just use the best loss.
However, early stopping is evaluated by MCC metrics.
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
from models.sp_tagging_bert import ProteinLMSequenceTaggingCRF
from models.sp_tagging_awd_lstm import ProteinAWDLSTMSequenceTaggingCRF
from models.sp_tagging_prottrans import XLNetSequenceTaggingCRF, ProteinXLNetTokenizer, BertSequenceTaggingCRF, ProteinBertTokenizer
from models.sp_tagging_unirep import UniRepSequenceTaggingCRF
from transformers import XLNetConfig, BertConfig
from models.awd_lstm import ProteinAWDLSTMConfig
from tape import visualization, ProteinBertConfig, UniRepConfig, TAPETokenizer
from train_scripts.utils.signalp_dataset import PartitionThreeLineFastaDataset
from torch.utils.data import DataLoader
from apex import amp

from train_scripts.downstream_tasks.sp_tagging_metrics_utils import get_discrepancy_rate

#import data
import os 
import wandb

from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score, recall_score, precision_score

#get the git hash - and log it
#wandb does that automatically - but only when in the correct directory when launching the job.
#by also doing it manually, force to launch from the correct directory, because otherwise this command will fail.
GIT_HASH = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode()

class Tokenizer(TAPETokenizer):
    '''Wrapper to enable from_pretrained() to load tokenizer, as in huggingface.'''
    def __init__(self, vocab: str = 'iupac', **kwargs):
        super().__init__(vocab)
   
    @classmethod
    def from_pretrained(cls,vocab, **kwargs):
        return cls(vocab, **kwargs)

MODEL_DICT = {
              'bert':   (ProteinBertConfig, ProteinLMSequenceTaggingCRF ),
              'awdlstm': (ProteinAWDLSTMConfig, ProteinAWDLSTMSequenceTaggingCRF),
              'xlnet': (XLNetConfig, XLNetSequenceTaggingCRF),
              'unirep': (UniRepConfig, UniRepSequenceTaggingCRF),
              'bert_prottrans': (BertConfig, BertSequenceTaggingCRF),
             }
TOKENIZER_DICT = {
                  'bert':    (TAPETokenizer, 'iupac'),
                  'awdlstm': (TAPETokenizer, 'iupac'),
                  'xlnet':   (ProteinXLNetTokenizer, 'Rostlab/prot_xlnet'),
                  'unirep':  (Tokenizer, 'unirep'),
                  'bert_prottrans': (ProteinBertTokenizer, 'Rostlab/prot_bert')
                 }


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S")
c_handler.setFormatter(formatter)
logger.addHandler(c_handler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Found device {device.type}.')

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
                    pred_sequence_labels: np.ndarray) -> Dict[str, float]:
    '''Utility function to get metrics from model output'''
    true_cs = tagged_seq_to_cs_multiclass(true_sequence_labels)
    pred_cs = tagged_seq_to_cs_multiclass(pred_sequence_labels)
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

    metrics_dict['Discrepancy'], metrics_dict['Multi-classified'] = get_discrepancy_rate(pred_global_labels,pred_sequence_labels)
    return metrics_dict

def training_step(model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, global_targets: torch.Tensor, optimizer: torch.optim.Optimizer, 
                    args: argparse.ArgumentParser, i: int, input_mask = None) -> (float, tuple):
    '''Predict one minibatch and performs update step.
    Returns:
        loss: loss value of the minibatch
    '''
    data = data.to(device)
    if input_mask is not None:
        input_mask = input_mask.to(device)
    targets = targets.to(device)
    global_targets = global_targets.to(device)
    model.train()
    optimizer.zero_grad()

    loss, _, _, _ = model(data, 
                            global_targets = global_targets,
                            targets=  targets ,
                            input_mask = input_mask)

    if torch.cuda.is_available():
        with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
    else:
        loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    if args.clip: 
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    return loss.item()


def validate(model: torch.nn.Module, valid_data: DataLoader) -> float:
    '''Run over the validation data. Average loss over the full set.
    '''
    model.eval()

    all_targets = []
    all_global_targets = []
    all_global_probs = []
    all_pos_preds = []

    total_loss = 0
    for i, batch in enumerate(valid_data):
        data, targets, input_mask, global_targets = batch
        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)
        global_targets = global_targets.to(device)
        input_mask = None
        with torch.no_grad():
            loss, global_probs, pos_probs, pos_preds = model(data, global_targets = global_targets, targets=  targets, input_mask = input_mask)

        total_loss += loss.item()
        all_targets.append(targets.detach().cpu().numpy())
        all_global_targets.append(global_targets.detach().cpu().numpy())
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)

    metrics = report_metrics(all_global_targets,all_global_probs, all_targets, all_pos_preds)

    try:
        global_roc_curve = wandb.plots.ROC(all_global_targets, all_global_probs, ['NO_SP', 'SP','LIPO', 'TAT'])
    except: 
        global_roc_curve = np.nan #sometimes wandb roc fails for numeric reasons

    val_metrics = {'loss': total_loss / len(valid_data), **metrics }
    return (total_loss / len(valid_data)), val_metrics, global_roc_curve


def main_training_loop(args: argparse.ArgumentParser):
    if args.enforce_walltime == True:
        loop_start_time = time.time()
        logger.info('Started timing loop')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #Setup Model
    #TODO how to inject new num_labels here?
    logger.info(f'Loading pretrained model in {args.resume}')
    config = MODEL_DICT[args.model_architecture][0].from_pretrained(args.resume)
    #patch LM model config for new downstream task
    setattr(config, 'num_labels', 6)
    setattr(config, 'num_global_labels', 4)
    setattr(config, 'classifier_hidden_size', args.classifier_hidden_size)
    setattr(config, 'use_crf', True)
    setattr(config, 'global_label_loss_multiplier', args.global_label_loss_multiplier)

    setattr(config, 'lm_output_dropout', args.lm_output_dropout)
    setattr(config, 'lm_output_position_dropout', args.lm_output_position_dropout)

    setattr(config, 'use_rnn', args.use_rnn)
    if args.use_rnn == True: #rnn training way more expensive than MLP
        setattr(args, 'epochs', 200)
    model = MODEL_DICT[args.model_architecture][1].from_pretrained(args.resume, config = config)
    #TODO is this called vocab in HF too?
    tokenizer = TOKENIZER_DICT[args.model_architecture][0].from_pretrained(TOKENIZER_DICT[args.model_architecture][1], do_lower_case =False)

    #training logger
    logger.info(f'Loaded weights from {args.resume} for model {model.base_model_prefix}')
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    experiment_name = f"{args.experiment_name}_{args.test_partition}_{args.validation_partition}_{time_stamp}"
    viz = visualization.get(args.output_dir, experiment_name, local_rank = -1) #debug=args.debug) #this -1 means traning is not distributed, debug makes experiment dry run for wandb
    
    #setup data
    val_id = args.validation_partition
    test_id = args.test_partition
    train_ids = [0,1,2,3,4]
    train_ids.remove(val_id)
    train_ids.remove(test_id)
    logger.info(f'Training on {train_ids}, validating on {val_id}')
    #if this is true, does not use tokenizer.encode(), but converts sequence step by step without adding the CLS, SEP tokens
    special_tokens_flag = False if args.model_architecture == 'awdlstm' else True #custom lm does not need cls, sep. (has cls in training, but for seq tagging useless)
    train_data = PartitionThreeLineFastaDataset(args.data, tokenizer= tokenizer, partition_id = train_ids, add_special_tokens = special_tokens_flag )
    val_data = PartitionThreeLineFastaDataset(args.data, tokenizer = tokenizer, partition_id = [val_id], add_special_tokens = special_tokens_flag)
    logger.info(f'{len(train_data)} training sequences, {len(val_data)} validation sequences.')

    train_loader = DataLoader(train_data, batch_size =args.batch_size, collate_fn= train_data.collate_fn, shuffle = True)
    val_loader = DataLoader(val_data, batch_size =args.batch_size, collate_fn= train_data.collate_fn)
    logger.info(f'Data loaded. One epoch = {len(train_loader)} batches.')

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

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
    viz.log_config({'git commit ID': GIT_HASH})
    viz.log_config(model.config.to_dict())
    viz.watch(model)
    logger.info(f'Logging experiment as {experiment_name} to wandb/tensorboard')
        
    #keep track of best loss
    stored_loss = 100000000
    learning_rate_steps = 0
    num_epochs_no_improvement = 0
    global_step = 0
    best_MCC_globallabel = 0
    best_MCC_cleavagesite = 0
    best_mcc_sum = 0
    for epoch in range(1, args.epochs+1):
        logger.info(f'Starting epoch {epoch}')
        viz.log_metrics({'Learning Rate': optimizer.param_groups[0]['lr'] }, "train", global_step)

        epoch_start_time = time.time()
        start_time = time.time() #for lr update interval
        
        for i, batch in enumerate(train_loader):

            data, targets, mask, global_targets = batch
            loss = training_step(model, data, targets, global_targets, optimizer, args, i, mask)
            viz.log_metrics({'loss': loss}, "train", global_step)
            global_step += 1

        logger.info(f'Step {global_step}, Epoch {epoch}: validating for {len(val_loader)} Validation steps')
        
        val_loss, val_metrics, roc_curve = validate(model, val_loader)
        viz.log_metrics(val_metrics, "val", global_step)
        if epoch == args.epochs:
           viz.log_metrics({'Detection roc curve': roc_curve}, 'val', global_step)

        logger.info(f"Validation: MCC global {val_metrics['Detection MCC']}, MCC seq {val_metrics['CS MCC']}. Epochs without improvement: {num_epochs_no_improvement}. lr step {learning_rate_steps}")


        #additional checkpoint saving. unclear what really works best - maybe use two different checkpoints, one for each task?
        if args.save_multi_checkpoint == True:
            mcc_sum = val_metrics['Detection MCC'] + val_metrics['CS MCC']
            #summed mccs
            if mcc_sum > best_mcc_sum:
                model.save_pretrained(os.path.join(args.output_dir, 'best_mcc_sum'))
                best_mcc_sum = mcc_sum
            #total loss
            if val_loss < stored_loss:
                model.save_pretrained(os.path.join(args.output_dir, 'best_loss'))
                stored_loss = val_loss
            #best cs
            if val_metrics['CS MCC'] > best_MCC_cleavagesite:
                model.save_pretrained(os.path.join(args.output_dir, 'best_cs'))
            #best label
            if val_metrics['Detection MCC'] > best_MCC_globallabel:
                model.save_pretrained(os.path.join(args.output_dir, 'best_label'))


        #keep track of optimization targets
        if (val_metrics['Detection MCC'] <= best_MCC_globallabel) and (val_metrics['CS MCC'] <= best_MCC_cleavagesite):
            num_epochs_no_improvement += 1
        else:
            num_epochs_no_improvement = 0

        best_MCC_globallabel = max(val_metrics['Detection MCC'], best_MCC_globallabel)
        best_MCC_cleavagesite = max(val_metrics['CS MCC'], best_MCC_cleavagesite)

        # if val_loss <stored_loss # not sure which criterion works better in practice. this ensures that no metric can increase at the cost of the other.
        # round because observed during training run that detection reaches a plateau.
        # don't want to block saving a new checkpoint that has goood cs mcc just because new detection mcc is 0.000001 smaller than the other plateau values.
        if (round(val_metrics['Detection MCC'],3) >= round(best_MCC_globallabel,3)) and (val_metrics['CS MCC'] >= best_MCC_cleavagesite):
            model.save_pretrained(args.output_dir)
            #also save with apex
            if torch.cuda.is_available():
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict()
                    }
                torch.save(checkpoint, os.path.join(args.output_dir, 'amp_checkpoint.pt'))
                logger.info(f'New best model with loss {val_loss}, MCC global {best_MCC_globallabel}, MCC seq {best_MCC_cleavagesite}, Saving model, training step {global_step}')



        if (epoch>args.min_epochs) and  (num_epochs_no_improvement > 10) and learning_rate_steps == 0:
            logger.info('No improvement for 10 epochs, reducing learning rate to 1/10.')
            num_epochs_no_improvement = 0
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * args.lr_step
            learning_rate_steps = 1

        if (epoch>args.min_epochs) and  (num_epochs_no_improvement > 10):
            logger.info('No improvement for 10 epochs, ending training early.')
            logger.info(f'Best: MCC global {best_MCC_globallabel}, MCC seq {best_MCC_cleavagesite}')

            logger.info('Rerunning validation data on best checkpoint.')
            model = MODEL_DICT[args.model_architecture][1].from_pretrained(args.output_dir)
            model.to(device)
            val_loss, val_metrics, roc_curve = validate(model, val_loader)
            logger.info(f"Validation: MCC global {val_metrics['Detection MCC']}, MCC seq {val_metrics['CS MCC']}")
            viz.log_metrics({'final Detection MCC': val_metrics['Detection MCC'], 'final CS MCC': val_metrics['CS MCC']}, 'val', global_step)

            if args.save_multi_checkpoint:
                model = MODEL_DICT[args.model_architecture][1].from_pretrained(os.path.join(args.output_dir, 'best_mcc_sum'))
                model.to(device)
                val_loss, val_metrics, roc_curve = validate(model, val_loader)
                logger.info(f"Validation best mcc sum checkpoint: MCC global {val_metrics['Detection MCC']}, MCC seq {val_metrics['CS MCC']}")    

                model = MODEL_DICT[args.model_architecture][1].from_pretrained(os.path.join(args.output_dir, 'best_loss'))
                model.to(device)
                val_loss, val_metrics, roc_curve = validate(model, val_loader)
                logger.info(f"Validation best loss checkpoint: MCC global {val_metrics['Detection MCC']}, MCC seq {val_metrics['CS MCC']}") 

                model = MODEL_DICT[args.model_architecture][1].from_pretrained(os.path.join(args.output_dir, 'best_cs'))
                model.to(device)
                val_loss, val_metrics, roc_curve = validate(model, val_loader)
                logger.info(f"Validation best cs checkpoint: MCC global {val_metrics['Detection MCC']}, MCC seq {val_metrics['CS MCC']}") 

                model = MODEL_DICT[args.model_architecture][1].from_pretrained(os.path.join(args.output_dir, 'best_label'))
                model.to(device)
                val_loss, val_metrics, roc_curve = validate(model, val_loader)
                logger.info(f"Validation best label checkpoint: MCC global {val_metrics['Detection MCC']}, MCC seq {val_metrics['CS MCC']}") 
            return (val_loss, best_MCC_globallabel, best_MCC_cleavagesite)            

        if  args.enforce_walltime == True and (time.time() - loop_start_time) > 84600: #23.5 hours
            logger.info('Wall time limit reached, ending training early')
            logger.info(f'Best: MCC global {best_MCC_globallabel}, MCC seq {best_MCC_cleavagesite}')
            return (val_loss, best_MCC_globallabel, best_MCC_cleavagesite)



    logger.info(f'Epoch {epoch}, epoch limit reached. Training complete')

    logger.info('Rerunning validation data on best checkpoint.')
    model = MODEL_DICT[args.model_architecture][1].from_pretrained(args.output_dir)
    model.to(device)
    val_loss, val_metrics, roc_curve = validate(model, val_loader)
    logger.info(f"Validation: MCC global {val_metrics['Detection MCC']}, MCC seq {val_metrics['CS MCC']}")
    viz.log_metrics({'final Detection MCC': val_metrics['Detection MCC'], 'final CS MCC': val_metrics['CS MCC']}, 'val', global_step)


    return (val_loss, best_MCC_globallabel, best_MCC_cleavagesite)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train CRF on top of Pfam Bert')
    parser.add_argument('--data', type=str, default='../data/data/signalp_5_data/',
                        help='location of the data corpus. Expects test, train and valid .fasta')
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
    parser.add_argument('--enforce_walltime', type=bool, default =True,
                        help='Report back current result before 24h wall time is over')
    parser.add_argument('--override_run_name', action = 'store_true',
                        help = 'override name with timestamp, save with split identifiers. Use when making checkpoints for crossvalidation.')
    parser.add_argument('--save_multi_checkpoint', action = 'store_true',
                        help = 'save checkpoints at additional criterions.')

    parser.add_argument('--global_label_loss_multiplier', type=float, default = 1.0,
                        help='multiplier for the crossentropy loss of the global label prediction. Use for sequence tagging/ global label performance tradeoff')
    parser.add_argument('--lm_output_dropout', type=float, default = 0.1,
                        help = 'dropout applied to LM output')
    parser.add_argument('--lm_output_position_dropout', type=float, default = 0.1,
                        help='dropout applied to LM output, drops full hidden states from sequence')

    #args for model architecture
    parser.add_argument('--model_architecture', type=str, default = 'bert',
                        help ='which model architecture the checkpoint is for')
    parser.add_argument('--use_rnn', action='store_true',
                        help='use biLSTM instead of MLP for emissions')
    parser.add_argument('--classifier_hidden_size', type=int, default=128, metavar='N',
                        help='Hidden size of the classifier head MLP')



    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #make unique output dir in output dir
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    full_name = '_'.join([args.experiment_name, 'test', str(args.test_partition), 'valid', str(args.validation_partition), time_stamp])
    
    if args.override_run_name == True:
        full_name ='_'.join([args.experiment_name, 'test', str(args.test_partition), 'valid', str(args.validation_partition)])

    args.output_dir = os.path.join(args.output_dir, full_name)
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