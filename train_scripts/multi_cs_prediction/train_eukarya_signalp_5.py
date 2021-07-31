'''
Train SignalP 5.0 architecture on new dataset, with more SP classes.
'''
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
from models.signalp_5 import SignalPConfig, SignalP5Model
from train_scripts.utils.signalp_dataset import SignalP5MultiCleavageSiteDataset, SIGNALP_KINGDOM_DICT
from torch.utils.data import DataLoader

from train_scripts.downstream_tasks.metrics_utils import get_metrics

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
        print('Decoy Wandb initiated, override wandb with no-op logging to prevent errors.')
        pass
    def log(self, value_dict, *args, **kwargs):
        #TODO should filter for train logs here, don't want to print at every step
        if list(value_dict.keys())[0].startswith('Train'):
            pass
        else:
            print(value_dict)
            print(args)
            print(kwargs)
    def watch(self, *args, **kwargs):
        pass
    

#get the git hash - and log it
#wandb does that automatically - but only when in the correct directory when launching the job.
#by also doing it manually, force to launch from the correct directory, because otherwise this command will fail.
GIT_HASH = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode()


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


def tagged_seq_to_multi_cs(tagged_seqs: np.ndarray, cs_tokens=[5]) -> List[List[int]]:
    '''Convert a sequences of tokens to the indices of the cleavage sites.
    Inputs:
        tagged_seqs: (batch_size, seq_len) integer array of position-wise labels
        sp_tokens: label tokens that indicate a signal peptide
    Returns:
        cs_sites: Nested list of predicted CS per sample.
    ''' 
    def get_cs_indices(x: np.ndarray) -> List[int]:
        cs_inds = np.where(np.isin(x, cs_tokens))[0] #all indices
        if len(cs_inds)>0:
            return (cs_inds+1).tolist()
        else:
            return [np.nan]

    out_list = []
    for sample_idx in range(tagged_seqs.shape[0]):
        sample_cs = get_cs_indices(tagged_seqs[sample_idx,:])
        out_list.append(sample_cs)

    return out_list

def tagged_seq_to_cs_multiclass(tagged_seqs: np.ndarray, sp_tokens = [9,10,11]):
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


# TODO nan handling of this process?
def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def repackage_cleavage_sites(true_cs: List[List[int]],pred_cs: List[List[int]], kingdom_ids: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    In multi-CS prediction, true_cs and pred_cs can be of different lengths.
    To solve this mismatch for metric calculation, recycle closest matching entries
    of the shorter of the two to match the length of the longer one.

    '''
    true_cs_flattened = []
    pred_cs_flattened = []
    kingdom_ids_out = []

    for i, (true, pred) in enumerate(zip(true_cs, pred_cs)):
        k = kingdom_ids[i]
        #iterate, recycle closest match in other list
        for idx in range(max(len(pred), len(true))):
            
            if idx < len(true):
                true_cs_flattened.append(true[idx])
            else:
                # more predicted CS than true CS. Repeat closest true CS
                best_match = _find_nearest(true, pred[idx])
                true_cs_flattened.append(best_match)

            if idx < len(pred):
                pred_cs_flattened.append(pred[idx])
            else:
                # more true CS than predicted CS. Repeat closest predicted CS
                best_match = _find_nearest(pred, true[idx])
                pred_cs_flattened.append(best_match)

            kingdom_ids_out.append(k)
            
    return np.array(true_cs_flattened), np.array(pred_cs_flattened), np.array(kingdom_ids_out)



def report_metrics_kingdom_averaged(true_global_labels: np.ndarray, pred_global_labels: np.ndarray, true_sequence_labels: np.ndarray, 
                    pred_sequence_labels: np.ndarray, kingdom_ids: np.ndarray, input_token_ids: np.ndarray, sp_tokens = [5],
                    true_cs = None
                    ) -> Dict[str, float]:
    '''Utility function to get metrics from model output'''

    if true_cs is None:
        true_cs = tagged_seq_to_multi_cs(true_sequence_labels, cs_tokens = sp_tokens)
        
    pred_cs = tagged_seq_to_multi_cs(pred_sequence_labels, cs_tokens = sp_tokens)

    true_cs, pred_cs, kingdom_ids_for_cs = repackage_cleavage_sites(true_cs, pred_cs, kingdom_ids)
    


    cs_kingdom = kingdom_ids_for_cs[~np.isnan(true_cs)]

    # TODO that doesn't work with multi-CS.
    # metric itself works, as it takes lists of CS and doesn't care about samples themselves.
    # Need to flatten pred and true CS list to same length. 
    # Add missed CS as np.nan to pred_CS
    # Add wrong CS as np.nan to true_CS
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
    all_token_ids = []
    all_cs = []
    total_loss = 0
    for i, batch in enumerate(train_data):

        if args.sp_region_labels:
            data, targets, input_mask, global_targets, kingdom_ids, cs = batch     
            all_cs.append(cs.numpy())
        else:
            data, targets, input_mask, global_targets, kingdom_ids = batch


        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)
        global_targets = global_targets.to(device)
        kingdom_ids = kingdom_ids.to(device)

        optimizer.zero_grad()

        loss, global_probs, pos_probs, pos_preds = model(data, 
                                                    global_targets = global_targets if not args.use_signalp6_labels else None,
                                                    targets_bitmap = targets if args.sp_region_labels else None, 
                                                    targets=  targets if not args.sp_region_labels else None,
                                                    input_mask = input_mask,
                                                    kingdom_ids = kingdom_ids
                                                    )
        loss = loss.mean() #if DataParallel because loss is a vector, if not doesn't matter
        #print(loss)
        #if torch.isnan(loss).any():
        #    import ipdb; ipdb.set_trace()

        total_loss += loss.item()
        all_targets.append(targets.detach().cpu().numpy())
        all_global_targets.append(global_targets.detach().cpu().numpy())
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())
        all_kingdom_ids.append(kingdom_ids.detach().cpu().numpy())
        all_token_ids.append(data.detach().cpu().numpy())

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

    
    all_targets = np.concatenate(all_targets)
    all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)
    all_kingdom_ids = np.concatenate(all_kingdom_ids)
    all_token_ids = np.concatenate(all_token_ids)
    all_cs = np.concatenate(all_cs) if args.sp_region_labels else None

    if args.use_signalp6_labels:
        sp_tokens = [5, 11, 19, 26, 31] if args.sp_region_labels else [3, 7, 11, 15,19]
    else:
        sp_tokens =  [5,8,13,17,19] if args.sp_region_labels else [5]
    metrics = report_metrics_kingdom_averaged(all_global_targets, all_global_probs, all_targets, all_pos_preds, all_kingdom_ids, all_token_ids, sp_tokens, all_cs)

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
    all_token_ids = []
    all_cs = []

    total_loss = 0
    for i, batch in enumerate(valid_data):

        if args.sp_region_labels:
            data, targets, input_mask, global_targets, kingdom_ids, cs = batch     
            all_cs.append(cs.numpy())
        else:
            data, targets, input_mask, global_targets, kingdom_ids = batch       
        
        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)
        global_targets = global_targets.to(device)
        kingdom_ids = kingdom_ids.to(device)

        with torch.no_grad():
            loss, global_probs, pos_probs, pos_preds = model(data, 
                                                            global_targets = global_targets if not args.use_signalp6_labels else None,
                                                            targets_bitmap = targets if args.sp_region_labels else None, 
                                                            targets=  targets if not args.sp_region_labels else None,
                                                            input_mask = input_mask,
                                                            kingdom_ids = kingdom_ids
                                                            )

        total_loss += loss.mean().item()
        all_targets.append(targets.detach().cpu().numpy())
        all_global_targets.append(global_targets.detach().cpu().numpy())
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())
        all_kingdom_ids.append(kingdom_ids.detach().cpu().numpy())
        all_token_ids.append(data.detach().cpu().numpy())


    all_targets = np.concatenate(all_targets)
    all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)
    all_kingdom_ids = np.concatenate(all_kingdom_ids)
    all_token_ids = np.concatenate(all_token_ids)
    all_cs = np.concatenate(all_cs) if args.sp_region_labels else None


    if args.use_signalp6_labels:
        sp_tokens = [5, 11, 19, 26, 31] if args.sp_region_labels else [3, 7, 11, 15,19]
    else:
        sp_tokens =  [5,8,13,17,19] if args.sp_region_labels else [5]
    
    metrics = report_metrics_kingdom_averaged(all_global_targets, all_global_probs, all_targets, all_pos_preds, all_kingdom_ids, all_token_ids, sp_tokens, all_cs)


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


    # Set seed
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        seed = args.random_seed
    else:
        seed = torch.seed()

    logger.info(f'torch seed: {seed}')
    wandb.config.update({'seed': seed})

    logger.info(f'Saving to {args.output_dir}')

    #Setup Model

    config = SignalPConfig(n_filters=args.n_filters,
                           filter_size=args.filter_size,
                           n_kingdoms=1,
                           hidden_size=args.hidden_size,
                           dropout_conv1=args.dropout_conv1,
                           dropout_input=args.dropout_input,
                           num_labels=args.num_seq_labels,
                           num_global_labels=args.num_global_labels,
                           num_layers=1)



    model = SignalP5Model(config)
    #wandb.watch(model) #still broken

    
    #setup data
    val_id = args.validation_partition
    test_id = args.test_partition
    train_ids = [0,1,2,3,4]
    train_ids.remove(val_id)
    train_ids.remove(test_id)
    logger.info(f'Training on {train_ids}, validating on {val_id}')


    kingdoms = ['EUKARYA']

    if args.sp_region_labels:
        train_data = SignalP5MultiCleavageSiteDataset(args.data,partition_id =train_ids)
        val_data = SignalP5MultiCleavageSiteDataset(args.data,partition_id =train_ids)
    else:
        train_data = SignalP5MultiCleavageSiteDataset(args.data,partition_id =train_ids)
        val_data = SignalP5MultiCleavageSiteDataset(args.data,partition_id =[val_id])

    logger.info(f'{len(train_data)} training sequences, {len(val_data)} validation sequences.')

    train_loader = DataLoader(train_data, batch_size =args.batch_size, collate_fn= train_data.collate_fn, shuffle = True)
    val_loader = DataLoader(val_data, batch_size =args.batch_size, collate_fn= train_data.collate_fn)

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
        #logger.info(f'Starting epoch {epoch}')

        
        epoch_loss, global_step = train(model, train_loader,optimizer,args,global_step)

        #logger.info(f'Step {global_step}, Epoch {epoch}: validating for {len(val_loader)} Validation steps')
        val_loss, val_metrics = validate(model, val_loader, args)
        log_metrics(val_metrics, "val", global_step)
        logger.info(f"Validation {epoch}: Loss {val_metrics['loss']}, MCC global {val_metrics['Detection MCC']}, MCC seq {val_metrics['CS MCC']}. Epochs without improvement: {num_epochs_no_improvement}. lr step {learning_rate_steps}")

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


    logger.info(f'Epoch {epoch}, epoch limit reached. Training complete')
    logger.info(f'Best: MCC Sum {best_mcc_sum}, Detection {best_mcc_global}, CS {best_mcc_cs}')
    log_metrics({'Best MCC Detection': best_mcc_global, 'Best MCC CS': best_mcc_cs, 'Best MCC sum': best_mcc_sum}, "val", global_step)

    print_all_final_metrics = False 
    #TODO get_metrics doesn't work with multi CS

    if print_all_final_metrics == True and best_mcc_sum>0: #check that there was a checkpoint that was saved.
        #reload best checkpoint
        model = SignalP5Model.from_pretrained(args.output_dir)

        ds = SignalP5MultiCleavageSiteDataset(args.data,partition_id =[test_id],kingdom_id=kingdoms)
        dataloader = torch.utils.data.DataLoader(ds, collate_fn = ds.collate_fn, batch_size = 80)
        metrics = get_metrics(model,dataloader, sp_tokens= [5,8,13,17,19] if args.sp_region_labels else [5])
        val_metrics = get_metrics(model,val_loader, sp_tokens= [5,8,13,17,19] if args.sp_region_labels else [5])

        if args.crossval_run or args.log_all_final_metrics:
            log_metrics(metrics, "test", global_step)
            log_metrics(val_metrics, "best_val", global_step)
        logger.info(metrics)
        logger.info('Validation set')
        logger.info(val_metrics)

        ## prettyprint everythingh
        import pandas as pd
        #df = pd.DataFrame.from_dict(x, orient='index')
        #df.index = df.index.str.split('_', expand=True)
        # print(df.sort_index()) 

        df = pd.DataFrame.from_dict([metrics,val_metrics]).T
        df.columns = ['test','val']
        df.index = df.index.str.split('_', expand=True)
        pd.set_option('display.max_rows', None)
        print(df.sort_index())

    return best_mcc_global, best_mcc_cs #best_mcc_sum



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train SignalP5 model.')
    parser.add_argument('--data', type=str, default='data/signal_peptides/signalp_updated_data/signalp_6_train_set.fasta',
                        help='location of the data corpus. Expects test, train and valid .fasta')
    parser.add_argument('--test_partition', type = int, default = 0,
                        help = 'partition that will not be used in this training run')
    parser.add_argument('--validation_partition', type = int, default = 1,
                        help = 'partition that will be used for validation in this training run')
            
    #args relating to training strategy.
    parser.add_argument('--lr', type=float, default=0.005,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--wdecay', type=float, default=0,
                        help='weight decay applied to all weights')
    parser.add_argument('--optimizer', type=str,  default='sgd',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--output_dir', type=str,  default=f'{time.strftime("%Y-%m-%d",time.gmtime())}_awd_lstm_lm_pretraining',
                        help='path to save logs and trained model')

    parser.add_argument('--experiment_name', type=str,  default='SignalP5',
                        help='experiment name for logging')
    parser.add_argument('--crossval_run', action = 'store_true',
                        help = 'override name with timestamp, save with split identifiers. Use when making checkpoints for crossvalidation.')
    parser.add_argument('--log_all_final_metrics', action='store_true', help='log all final test/val metrics to w&b')



    parser.add_argument('--num_seq_labels', type=int, default=12)
    parser.add_argument('--num_global_labels', type=int, default=6)


    parser.add_argument('--crf_scaling_factor', type=float, default=1.0, help='Scale CRF NLL by this before adding to global label loss')
    parser.add_argument('--dropout_conv1', type=float, default=0.15)
    parser.add_argument('--dropout_input', type=float, default=0.25)
    parser.add_argument('--hidden_size', type=int, default=64, help='LSTM hidden size (bidirectional: final size = n*2')
    parser.add_argument('--n_filters',type=int, default=32, help='conv1 number of filters')
    parser.add_argument('--filter_size',type=int, default=3, help='conv1 kernel size')

    #use random seeds, but avoid problem when torch.seed() returns a non-printable float
    # this causes run to fail, can't have that here, need all runs to pass
    parser.add_argument('--random_seed', type=int, default=np.random.randint(999999999999999))

    parser.add_argument('--sp_region_labels', action='store_true', help = 'Use region labels instead of standard signalp labels')
    parser.add_argument('--use_signalp6_labels', action='store_true', help = 'Use the CRF implementation of SP6+corresponding labels')



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
