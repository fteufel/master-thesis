'''
Use the pretrained Bert model to get embeddings using DCT.
Train a classifier on the frozen embeddings.

Script full of useless boilerplate code bc adapted from full model training loop.
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
from models.multi_crf_bert import ProteinBertTokenizer
from transformers import BertModel
from train_scripts.utils.signalp_dataset import LargeCRFPartitionDataset, SIGNALP_KINGDOM_DICT, RegionCRFDataset
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_scripts.downstream_tasks.metrics_utils import get_metrics_multistate, find_cs_tag, compute_mcc, mask_cs
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


from scipy.fftpack import dct
def dct_embeddings(seqs, k_to_keep =3):
    outlist= []
    #one by one dct, as in code from paper https://github.com/N-Almarwani/DCT_Sentence_Embedding/blob/master/DCT.py
    for seq in seqs:
        transformed = np.reshape(dct(seq, norm='ortho', axis=0)[:k_to_keep,:], (k_to_keep*seq.shape[1],))
        outlist.append(transformed)

    return np.array(outlist)

REVERSE_GLOBAL_LABEL_DICT = {0 : 'NO_SP', 1: 'SP', 2: 'LIPO', 3: 'TAT', 4: 'TATLIPO', 5:'PILIN'}

def compute_metrics(all_global_targets: np.ndarray, all_global_preds: np.ndarray, 
                    all_kingdom_ids: List[str],
                    rev_label_dict: Dict[int,str] = REVERSE_GLOBAL_LABEL_DICT): 
    '''Compute the metrics used in the SignalP 5.0 supplement.
    Returns all metrics as a dict. 
    '''    
    rev_kingdom_dict = dict(zip(SIGNALP_KINGDOM_DICT.values(), SIGNALP_KINGDOM_DICT.keys()))
    all_kingdom_ids = np.array([rev_kingdom_dict[x] for x in all_kingdom_ids])
    metrics_dict = {}

    for kingdom in np.unique(all_kingdom_ids):
        #subset for the kingdom
        kingdom_targets = all_global_targets[all_kingdom_ids == kingdom] 
        kingdom_preds = all_global_preds[all_kingdom_ids == kingdom]


        for sp_type in np.unique(kingdom_targets):
            #subset for each sp type
            if sp_type == 0:
                continue #skip NO_SP

            #use full data
            metrics_dict[f'{kingdom}_{rev_label_dict[sp_type]}_mcc2'] = compute_mcc(kingdom_targets,kingdom_preds,label_positive=sp_type)

            #subset for no_sp
            mcc1_idx = np.isin(kingdom_targets, [sp_type, 0])
            metrics_dict[f'{kingdom}_{rev_label_dict[sp_type]}_mcc1'] = compute_mcc(kingdom_targets[mcc1_idx], kingdom_preds[mcc1_idx], label_positive=sp_type)

    return metrics_dict


def report_metrics(true_global_labels: np.ndarray, pred_global_labels: np.ndarray, kingdom_ids: np.ndarray =None) -> Dict[str, float]:
    '''Utility function to get metrics from model output'''

    #applying a threhold of 0.25 (SignalP) to a 4 class case is equivalent to the argmax.
    pred_global_labels_thresholded = pred_global_labels.argmax(axis=1)
    metrics_dict = {}
    metrics_dict['Detection MCC'] = matthews_corrcoef(true_global_labels, pred_global_labels_thresholded)

    #add the real metrics (the one above is for the stopping criterion)
    metrics = compute_metrics(true_global_labels,pred_global_labels_thresholded,kingdom_ids)
    metrics_dict.update(metrics)

    return metrics_dict

def report_metrics_kingdom_averaged(true_global_labels: np.ndarray, pred_global_labels: np.ndarray, kingdom_ids: np.ndarray) -> Dict[str, float]:
    '''Utility function to get metrics from model output'''

    #applying a threhold of 0.25 (SignalP) to a 4 class case is equivalent to the argmax.
    pred_global_labels_thresholded = pred_global_labels.argmax(axis=1)

    #compute metrics for each kingdom
    rev_kingdom_dict = dict(zip(SIGNALP_KINGDOM_DICT.values(), SIGNALP_KINGDOM_DICT.keys()))
    all_detection_mcc = []
    metrics_dict = {}
    for kingdom in np.unique(kingdom_ids):
        kingdom_global_labels = true_global_labels[kingdom_ids==kingdom] 
        kingdom_pred_global_labels_thresholded = pred_global_labels_thresholded[kingdom_ids==kingdom]

        metrics_dict[f'Detection MCC {rev_kingdom_dict[kingdom]}'] = matthews_corrcoef(kingdom_global_labels, kingdom_pred_global_labels_thresholded)

        all_detection_mcc.append(metrics_dict[f'Detection MCC {rev_kingdom_dict[kingdom]}'])

    metrics_dict['Detection MCC'] = sum(all_detection_mcc)/len(all_detection_mcc)
    #add the real metrics (the one above is for the stopping criterion)
    metrics = compute_metrics(true_global_labels,pred_global_labels_thresholded,kingdom_ids)
    metrics_dict.update(metrics)

    return metrics_dict


def train(model: torch.nn.Module, pred_model: torch.nn.Module, train_data: DataLoader, optimizer: torch.optim.Optimizer, 
                    args: argparse.ArgumentParser, global_step: int) -> Tuple[float, int]:
    '''Predict one minibatch and performs update step.
    Returns:
        loss: loss value of the minibatch
    '''

    model.train()
    optimizer.zero_grad()

    all_global_targets = []
    all_global_probs = []
    all_kingdom_ids = [] #gather ids for kingdom-averaged metrics
    total_loss = 0
    for i, batch in enumerate(train_data):
        if args.sp_region_labels:
            data, targets, input_mask, global_targets, cleavage_sites, sample_weights, kingdom_ids = batch        
        else:
            data, targets, input_mask, global_targets, sample_weights, kingdom_ids = batch

        data = data.to(device)
        input_mask = input_mask.to(device)
        global_targets = global_targets.to(device)
        kingdom_ids = kingdom_ids.to(device) 

        with torch.no_grad():
            hidden_state, _ = model(data, input_mask)
            embeds = dct_embeddings(hidden_state.cpu().numpy(), 3)
            embeds = torch.tensor(embeds).to(device)

        #linear layer on top
        global_logits = pred_model(embeds)
        loss = nn.functional.cross_entropy(global_logits,global_targets)
        global_probs = nn.functional.softmax(global_logits, dim=1)

        loss = loss.mean() #if DataParallel because loss is a vector, if not doesn't matter
        total_loss += loss.item()
        all_global_targets.append(global_targets.detach().cpu().numpy())
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_kingdom_ids.append(kingdom_ids.detach().cpu().numpy())


        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        log_metrics({'loss': loss.item()}, "train", global_step)
         

        if args.optimizer == 'smart_adamax':
            log_metrics({'Learning rate': optimizer.get_lr()[0]}, "train", global_step)
        else:
            log_metrics({'Learning Rate': optimizer.param_groups[0]['lr'] }, "train", global_step)
        global_step += 1

    
    all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_kingdom_ids = np.concatenate(all_kingdom_ids)

    if args.average_per_kingdom:
        metrics = report_metrics_kingdom_averaged(all_global_targets,all_global_probs,all_kingdom_ids)
    else:
        metrics = report_metrics(all_global_targets,all_global_probs,all_kingdom_ids)
    log_metrics(metrics, 'train', global_step)


    return total_loss/len(train_data), global_step


def validate(model: torch.nn.Module, pred_model, valid_data: DataLoader, args) -> float:
    '''Run over the validation data. Average loss over the full set.
    '''
    model.eval()

    all_global_targets = []
    all_global_probs = []
    all_kingdom_ids = []

    total_loss = 0
    for i, batch in enumerate(valid_data):
        if args.sp_region_labels:
            data, targets, input_mask, global_targets, cleavage_sites, sample_weights, kingdom_ids = batch        
        else:
            data, targets, input_mask, global_targets, sample_weights, kingdom_ids = batch

        data = data.to(device)
        input_mask = input_mask.to(device)
        global_targets = global_targets.to(device)
        kingdom_ids = kingdom_ids.to(device) 

        with torch.no_grad():
            hidden_state, _ = model(data, input_mask)
            embeds = dct_embeddings(hidden_state.cpu().numpy(), 3)
            embeds = torch.tensor(embeds).to(device)
        with torch.no_grad():
            global_logits = pred_model(embeds)
            loss = nn.functional.cross_entropy(global_logits,global_targets)
            global_probs = nn.functional.softmax(global_logits, dim=1)

        total_loss += loss.mean().item()
        all_global_targets.append(global_targets.detach().cpu().numpy())
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_kingdom_ids.append(kingdom_ids.detach().cpu().numpy())



    all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_kingdom_ids = np.concatenate(all_kingdom_ids)

    if args.average_per_kingdom:
        metrics = report_metrics_kingdom_averaged(all_global_targets,all_global_probs,all_kingdom_ids)
    else:
        metrics = report_metrics(all_global_targets,all_global_probs,all_kingdom_ids)


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
    wandb.init(dir=args.output_dir, name=experiment_name)


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
    logger.info(f'Loading pretrained model in {args.resume}')



    model = BertModel.from_pretrained('Rostlab/prot_bert')
    tokenizer = ProteinBertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)

    #1024 *k_to_keep
    # 6 labels
    pred_model = nn.Linear(1024*3,6)


    #setup data
    val_id = args.validation_partition
    test_id = args.test_partition
    train_ids = [0,1,2]
    train_ids.remove(val_id)
    train_ids.remove(test_id)
    logger.info(f'Training on {train_ids}, validating on {val_id}')


    kingdoms = ['EUKARYA', 'ARCHAEA', 'NEGATIVE', 'POSITIVE']


    train_data = LargeCRFPartitionDataset(args.data, args.sample_weights ,tokenizer= tokenizer, partition_id = train_ids, kingdom_id=kingdoms, 
                                            add_special_tokens = True,return_kingdom_ids=True)
    val_data = LargeCRFPartitionDataset(args.data, args.sample_weights , tokenizer = tokenizer, partition_id = [val_id], kingdom_id=kingdoms, 
                                            add_special_tokens = True, return_kingdom_ids=True)        
    logger.info(f'{len(train_data)} training sequences, {len(val_data)} validation sequences.')

    train_loader = DataLoader(train_data, batch_size =args.batch_size, collate_fn= train_data.collate_fn, shuffle = True)
    val_loader = DataLoader(val_data, batch_size =args.batch_size, collate_fn= train_data.collate_fn)

    logger.info(f'Data loaded. One epoch = {len(train_loader)} batches.')

    #set up wandb logging, login and project id from commandline vars
    wandb.config.update(args)
    wandb.config.update({'git commit ID': GIT_HASH})
    wandb.watch(pred_model)
    logger.info(f'Logging experiment as {experiment_name} to wandb/tensorboard')
    logger.info(f'Saving checkpoints at {args.output_dir}')


    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(pred_model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(pred_model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adamax':
        optimizer = torch.optim.Adamax(pred_model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'smart_adamax':
        t_total = len(train_loader) * args.epochs
        optimizer = Adamax(pred_model.parameters(), 
                           lr = args.lr, 
                           warmup = 0.1,
                           t_total = t_total,
                           schedule='warmup_linear', 
                           betas = (0.9, 0.999),
                           weight_decay=args.wdecay,
                           max_grad_norm=1)



    model.to(device)
    pred_model.to(device)
    logger.info('Model set up!')

    logger.info(f'Running model on {device}, not using nvidia apex')

    save_file = os.path.join(args.output_dir, 'checkpoint.pt')
        
    #keep track of best loss
    stored_loss = 100000000
    learning_rate_steps = 0
    num_epochs_no_improvement = 0
    global_step = 0
    best_mcc_global = 0
    best_mcc_cs = 0
    for epoch in range(1, args.epochs+1):
        logger.info(f'Starting epoch {epoch}')

        
        epoch_loss, global_step = train(model, pred_model, train_loader,optimizer,args,global_step)

        logger.info(f'Step {global_step}, Epoch {epoch}: validating for {len(val_loader)} Validation steps')
        val_loss, val_metrics = validate(model, pred_model, val_loader, args)
        log_metrics(val_metrics, "val", global_step)
        logger.info(f"Validation: MCC global {val_metrics['Detection MCC']}. Epochs without improvement: {num_epochs_no_improvement}. lr step {learning_rate_steps}")

        if val_metrics['Detection MCC'] > best_mcc_global:
            best_mcc_global = val_metrics['Detection MCC']
            num_epochs_no_improvement = 0

            model.save_pretrained(args.output_dir)
            torch.save(pred_model.state_dict(), save_file)
            logger.info(f'New best model with loss {val_loss}, MCC global {val_metrics["Detection MCC"]}, training step {global_step}')

        else:
            num_epochs_no_improvement += 1




    logger.info(f'Epoch {epoch}, epoch limit reached. Training complete')
    logger.info(f'Best: Detection {best_mcc_global}')
    log_metrics({'Best MCC Detection': best_mcc_global}, "val", global_step)

    print_all_final_metrics = True #TODO get_metrics is not adapted yet to large crf
    if print_all_final_metrics == True:
        #reload best checkpoint
        #TODO Kingdom ID handling needs to be added here.
        #model = MODEL_DICT[args.model_architecture][1].from_pretrained(args.output_dir)
        pred_model.load_state_dict(torch.load(save_file))
        ds = LargeCRFPartitionDataset(args.data, args.sample_weights , tokenizer = tokenizer, partition_id = [test_id], kingdom_id=kingdoms, 
                                            add_special_tokens = True, return_kingdom_ids=True)
        dataloader = torch.utils.data.DataLoader(ds, collate_fn = ds.collate_fn, batch_size = 80)


        ## Run validate function on test set to get metrics
        loss, metrics = validate(model,pred_model,dataloader,args)
        loss, val_metrics = validate(model, pred_model, val_loader,args)

        if args.crossval_run or args.log_all_final_metrics:
            log_metrics(metrics, "test", global_step)
            log_metrics(val_metrics, "best_val", global_step)
        logger.info(metrics)
        logger.info('Validation set')
        logger.info(val_metrics)

        ## prettyprint everything
        import pandas as pd
        #df = pd.DataFrame.from_dict(x, orient='index')
        #df.index = df.index.str.split('_', expand=True)
        # print(df.sort_index()) 

        df = pd.DataFrame.from_dict([metrics,val_metrics]).T
        df.columns = ['test','val']
        df.index = df.index.str.split('_', expand=True)
        pd.set_option('display.max_rows', None)
        print(df.sort_index())

    run_completed=True
    return best_mcc_global, best_mcc_cs, run_completed #best_mcc_sum






if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train CRF on top of Pfam Bert')
    parser.add_argument('--data', type=str, default='data/signal_peptides/signalp_updated_data/signalp_6_train_set.fasta',
                        help='location of the data corpus. Expects test, train and valid .fasta')
    parser.add_argument('--sample_weights', type=str, default=None,
                        help='path to .csv file with the weights for each sample')
    parser.add_argument('--test_partition', type = int, default = 0,
                        help = 'partition that will not be used in this training run')
    parser.add_argument('--validation_partition', type = int, default = 1,
                        help = 'partition that will be used for validation in this training run')
            
    #args relating to training strategy.
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=5,
                        help='upper epoch limit')

    parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--optimizer', type=str,  default='adam',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--output_dir', type=str,  default=f'{time.strftime("%Y-%m-%d",time.gmtime())}_awd_lstm_lm_pretraining',
                        help='path to save logs and trained model')
    parser.add_argument('--resume', type=str,  default='bert-base',
                        help='path of model to resume (directory containing .bin and config.json')
    parser.add_argument('--experiment_name', type=str,  default='BERT-DCT',
                        help='experiment name for logging')
    parser.add_argument('--crossval_run', action = 'store_true',
                        help = 'override name with timestamp, save with split identifiers. Use when making checkpoints for crossvalidation.')
    parser.add_argument('--log_all_final_metrics', action='store_true', help='log all final test/val metrics to w&b')


    parser.add_argument('--average_per_kingdom', action='store_true',
                        help='Average MCCs per kingdom instead of overall computatition')
    parser.add_argument('--crf_scaling_factor', type=float, default=1.0, help='Scale CRF NLL by this before adding to global label loss')
    parser.add_argument('--use_weighted_kingdom_sampling', action='store_true',
                        help='upsample all kingdoms to equal probabilities')
    parser.add_argument('--random_seed', type=int, default=None, help='random seed for torch.')
    parser.add_argument('--additional_train_set', type=str,default=None, help='Additional samples to train on')

    #args for model architecture
    parser.add_argument('--sp_region_labels', action='store_true', help='Use labels for n,h,c regions of SPs.')
    parser.add_argument('--constrain_crf', action='store_true', help='Constrain the transitions of the region-tagging CRF.')


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