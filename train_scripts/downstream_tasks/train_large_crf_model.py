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
from models.sp_tagging_bert import ProteinLMSequenceTaggingCRF
from models.sp_tagging_awd_lstm import ProteinAWDLSTMSequenceTaggingCRF
from models.sp_tagging_prottrans import XLNetSequenceTaggingCRF, ProteinXLNetTokenizer, BertSequenceTaggingCRF, ProteinBertTokenizer
from models.sp_tagging_unirep import UniRepSequenceTaggingCRF
from transformers import XLNetConfig, BertConfig
from models.awd_lstm import ProteinAWDLSTMConfig
from tape import visualization, ProteinBertConfig, UniRepConfig, TAPETokenizer
from train_scripts.utils.signalp_dataset import LargeCRFPartitionDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from apex import amp

from train_scripts.downstream_tasks.metrics_utils import get_metrics

from train_scripts.utils.perturbation import SmartPerturbation
from train_scripts.utils.smart_optim import Adamax
from models.utils.mixout_utils import apply_mixout_to_xlnet


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
    true_cs = tagged_seq_to_cs_multiclass(true_sequence_labels, sp_tokens = [3,7,11])
    pred_cs = tagged_seq_to_cs_multiclass(pred_sequence_labels, sp_tokens = [3,7,11])
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

def train(model: torch.nn.Module, train_data: DataLoader, optimizer: torch.optim.Optimizer, 
                    args: argparse.ArgumentParser, global_step: int, visualizer, adversarial_teacher = None) -> Tuple[float, int]:
    '''Predict one minibatch and performs update step.
    Returns:
        loss: loss value of the minibatch
    '''

    model.train()
    optimizer.zero_grad()

    total_loss = 0
    for i, batch in enumerate(train_data):
        data, targets, input_mask, global_targets, sample_weights = batch
        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)
        sample_weights = sample_weights.to(device) if args.use_sample_weights else None

        loss, _, pos_probs, _ = model(data, 
                                targets=  targets ,
                                input_mask = input_mask,
                                sample_weights = sample_weights)

        total_loss += loss.item()
        #SMART proximal point optimization
        if adversarial_teacher:
        #TODO finish this, add new optimizer with momentum and lr scheduling. Then SMART implementation is complete
            log_probs = torch.exp(pos_probs)
            adv_loss = adversarial_teacher.forward(model, log_probs, input_ids =data, attention_mask = input_mask)
            visualizer.log_metrics({'Raw loss': loss}, "train", global_step)
            visualizer.log_metrics({'Regularization loss': adv_loss}, "train", global_step)

            loss = loss + args.adv_alpha * adv_loss

        if torch.cuda.is_available():
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        #from IPython import embed; embed()
        optimizer.step()

        visualizer.log_metrics({'loss': loss}, "train", global_step)
        #logger.info('optimizer state:')
        #logger.info(optimizer.state['step'])
        #import ipdb; ipdb.set_trace()
        #logger.info(f'optimizer lr {optimizer.get_lr()[0]}')
        if args.optimizer == 'smart_adamax':
            visualizer.log_metrics({'Learning rate': optimizer.get_lr()[0]}, "train", global_step)
        else:
            visualizer.log_metrics({'Learning Rate': optimizer.param_groups[0]['lr'] }, "train", global_step)
        global_step += 1

    #https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/main.py
    # do not add momentum for now.

    return total_loss/len(train_data), global_step


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
        data, targets, input_mask, global_targets, sample_weights = batch
        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)
        global_targets = global_targets.to(device)
        sample_weights = sample_weights.to(device) if args.use_sample_weights else None

        with torch.no_grad():
            loss, global_probs, pos_probs, pos_preds = model(data, 
                                                            targets=  targets, 
                                                            #global_targets = global_targets, 
                                                            sample_weights=sample_weights,
                                                            input_mask = input_mask)

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


    val_metrics = {'loss': total_loss / len(valid_data), **metrics }
    return (total_loss / len(valid_data)), val_metrics

def main_training_loop(args: argparse.ArgumentParser):


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #Setup Model
    #TODO how to inject new num_labels here?
    logger.info(f'Loading pretrained model in {args.resume}')
    config = MODEL_DICT[args.model_architecture][0].from_pretrained(args.resume)
    #patch LM model config for new downstream task
    setattr(config, 'num_labels', 7 if args.eukarya_only else 15)
    setattr(config, 'num_global_labels', 2 if args.eukarya_only else 4)
    setattr(config, 'classifier_hidden_size', args.classifier_hidden_size)
    setattr(config, 'use_crf', True)

    setattr(config, 'lm_output_dropout', args.lm_output_dropout)
    setattr(config, 'lm_output_position_dropout', args.lm_output_position_dropout)
    setattr(config, 'use_large_crf', True)

    if args.remove_top_layer:
        setattr(config, 'n_layer', config.n_layer-1)

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
    kingdoms = ['EUKARYA'] if args.eukarya_only else ['EUKARYA', 'ARCHAEA', 'NEGATIVE', 'POSITIVE']
    special_tokens_flag = False if args.model_architecture == 'awdlstm' else True #custom lm does not need cls, sep. (has cls in training, but for seq tagging useless)
    train_data = LargeCRFPartitionDataset(args.data, args.sample_weights ,tokenizer= tokenizer, partition_id = train_ids, kingdom_id=kingdoms, add_special_tokens = special_tokens_flag )
    val_data = LargeCRFPartitionDataset(args.data, args.sample_weights , tokenizer = tokenizer, partition_id = [val_id], kingdom_id=kingdoms, add_special_tokens = special_tokens_flag)
    logger.info(f'{len(train_data)} training sequences, {len(val_data)} validation sequences.')

    train_loader = DataLoader(train_data, batch_size =args.batch_size, collate_fn= train_data.collate_fn, shuffle = True)
    val_loader = DataLoader(val_data, batch_size =args.batch_size, collate_fn= train_data.collate_fn)

    if args.use_random_weighted_sampling:
        sampler = WeightedRandomSampler(train_data.sample_weights, len(train_data), replacement=False)
        train_loader = DataLoader(train_data, batch_size = args.batch_size, collate_fn = train_data.collate_fn, sampler = sampler)

    logger.info(f'Data loaded. One epoch = {len(train_loader)} batches.')

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'smart_adamax':
        t_total = len(train_loader) * args.epochs
        optimizer = Adamax(model.parameters(), lr = args.lr, warmup = 0.1,t_total = t_total,schedule='warmup_linear', betas = (0.9, 0.999),max_grad_norm=1)

    if args.use_smart_perturbation:
        logger.info('Using adversarial perturbation regularization')
        adversarial_teacher = SmartPerturbation(epsilon=1e-5,step_size=0.001,noise_var=1e-5,k=1,norm_level=0)
    else:
        adversarial_teacher = None

    if args.use_mixout:
        apply_mixout_to_xlnet(model, 0.9)

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
    best_mcc_sum = 0
    for epoch in range(1, args.epochs+1):
        logger.info(f'Starting epoch {epoch}')
        #viz.log_metrics({'Learning Rate': optimizer.param_groups[0]['lr'] }, "train", global_step)
        #viz.log_metrics({'Learning Rate': optimizer.get_lr()[0]}, 'train', global_step)

        epoch_start_time = time.time()
        start_time = time.time() #for lr update interval
        
        epoch_loss, global_step = train(model, train_loader,optimizer,args,global_step,viz,adversarial_teacher)

        logger.info(f'Step {global_step}, Epoch {epoch}: validating for {len(val_loader)} Validation steps')
        val_loss, val_metrics = validate(model, val_loader)
        viz.log_metrics(val_metrics, "val", global_step)
        logger.info(f"Validation: MCC global {val_metrics['Detection MCC']}, MCC seq {val_metrics['CS MCC']}. Epochs without improvement: {num_epochs_no_improvement}. lr step {learning_rate_steps}")

        mcc_sum = val_metrics['Detection MCC'] + val_metrics['CS MCC']
        if mcc_sum > best_mcc_sum:
            best_mcc_sum = mcc_sum
            num_epochs_no_improvement = 0
            model.save_pretrained(args.output_dir)
            logger.info(f'New best model with loss {val_loss}, MCC global {val_metrics["Detection MCC"]}, MCC seq {val_metrics["CS MCC"]}, Saving model, training step {global_step}')

        else:
            num_epochs_no_improvement += 1


        #if (epoch>args.min_epochs) and  (num_epochs_no_improvement > 10) and learning_rate_steps == 0:
        #    logger.info('No improvement for 10 epochs, reducing learning rate to 1/10.')
        #    num_epochs_no_improvement = 0
        #    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
        #    learning_rate_steps = 1


        if (epoch>args.min_epochs) and  (num_epochs_no_improvement > 10):
            logger.info('No improvement for 10 epochs, ending training early.')
            logger.info(f'Best: MCC Sum {best_mcc_sum}')

            return best_mcc_sum

    logger.info(f'Epoch {epoch}, epoch limit reached. Training complete')
    logger.info(f'Best: MCC Sum {best_mcc_sum}')

    print_all_final_metrics = True
    if print_all_final_metrics == True:
        #reload best checkpoint
        model = MODEL_DICT[args.model_architecture][1].from_pretrained(args.output_dir)
        ds = LargeCRFPartitionDataset(args.data, tokenizer= tokenizer, partition_id = [test_id], 
                                      kingdom_id=kingdoms, add_special_tokens = special_tokens_flag)
        dataloader = torch.utils.data.DataLoader(ds, collate_fn = ds.collate_fn, batch_size = 80)
        metrics = get_metrics(model,dataloader)
        logger.info(metrics)

    return best_mcc_sum



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train CRF on top of Pfam Bert')
    parser.add_argument('--data', type=str, default='../data/data/signalp_5_data/',
                        help='location of the data corpus. Expects test, train and valid .fasta')
    parser.add_argument('--sample_weights', type=str, default='../data/data/sample_weights.csv',
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
    parser.add_argument('--override_run_name', action = 'store_true',
                        help = 'override name with timestamp, save with split identifiers. Use when making checkpoints for crossvalidation.')

    parser.add_argument('--eukarya_only', action='store_true', help = 'Only train on eukarya SPs')


    parser.add_argument('--lm_output_dropout', type=float, default = 0.1,
                        help = 'dropout applied to LM output')
    parser.add_argument('--lm_output_position_dropout', type=float, default = 0.1,
                        help='dropout applied to LM output, drops full hidden states from sequence')
    parser.add_argument('--use_smart_perturbation', action='store_true')
    parser.add_argument('--adv_alpha', type=float, default =1,
                        help ='weight of the adversarial regularization loss')
    parser.add_argument('--use_mixout', action='store_true',
                        help='Apply mixout regularization to the model. config.dropout is used as mixout rate')
    parser.add_argument('--use_sample_weights', action = 'store_true', 
                        help = 'Use sample weights to rescale loss per sample')
    parser.add_argument('--use_random_weighted_sampling', action='store_true',
                        help='use sample weights to load random samples as minibatches according to weights')

    #args for model architecture
    parser.add_argument('--model_architecture', type=str, default = 'bert',
                        help ='which model architecture the checkpoint is for')
    parser.add_argument('--use_rnn', action='store_true',
                        help='use biLSTM instead of MLP for emissions')
    parser.add_argument('--classifier_hidden_size', type=int, default=128, metavar='N',
                        help='Hidden size of the classifier head MLP')
    parser.add_argument('--remove_top_layer', action='store_true', 
                        help='Remove the top layer of the LM')



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