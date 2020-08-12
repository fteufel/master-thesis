# Script to fine-tune a pretrained AWD-LSTM model.
#like normal training loop, but optionally uses slanted triangular learning rate schedule
#Felix July 2020
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
from models.awd_lstm import ProteinAWDLSTMConfig
from models.sp_tagging_awd_lstm import ProteinAWDLSTMPointerSentinelModel
from tape import visualization #import utils.visualization as visualization
from utils.signalp_dataset import PointerSentinelThreeLineFastaDataset
from torch.utils.data import DataLoader
from apex import amp
import wandb 
import os 

from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score

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


def training_step(model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, optimizer: torch.optim.Optimizer, 
                    args: argparse.ArgumentParser, i: int) -> (float, tuple):
    '''Predict one minibatch and performs update step.
    Returns:
        loss: loss value of the minibatch
    '''
    data = data.to(device)
    targets = targets.to(device)


    model.train()
    optimizer.zero_grad()


    loss, _ = model(data, targets= targets)

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

def compute_cs_detection(outputs: np.array, targets: np.array, window_size: int = 1):
    '''Calculate CS detection metrics.
    '''
    predictions_all = outputs.argmax(axis =1) #don't define threshold yet, the one with max score is the label for the position.
    #split by sentinel - negative samples don't get the window.
    negative_samples_idx = (targets == 70)
    positive_samples_idx = (targets != 70)

    predictions_pos = predictions_all[positive_samples_idx]
    targets_pos = targets[positive_samples_idx]

    #check if prediction is within +-window_size of target
    correct_pos_preds = (predictions_pos >= targets_pos -window_size) & (predictions_pos<= targets_pos+window_size)
    true_pos = correct_pos_preds.sum()
    false_neg =  len(correct_pos_preds) - correct_pos_preds.sum()
    #negatives - no window
    false_pos = (predictions_all[negative_samples_idx] != 70).sum()
    true_neg = (predictions_all[negative_samples_idx] ==70).sum()
    recall =  true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)

    return precision, recall

def compute_cs_detection_no_threshold(outputs: np.array, targets: np.array, window_size: int = 1):
    '''Calculate threshold-free CS detection metrics.
    Score for CS detection is the sum of the probabilities in the window
    Problem is framed as binary classification: true label 1: has SP, 0: no SP
    Scores for positive samples: Sum of probabilities within window
    Scores for negative samples: Sum of probabilites over whole sequence without the sentinel
    NOTE sentinel position idx (70) is hardcoded  
    '''
    # outputs are position-wise probabilities of shape batch_size, seq_len
    # from each sequence, sum over the positions within the window
    window_borders_low = targets - window_size
    window_borders_high = targets + window_size +1

    def get_window_sum(data, lower_indices, higher_indices):
        window_sums = [data[i, low:high].sum() for i, (low, high) in enumerate(zip(lower_indices, higher_indices))]
        return np.array(window_sums)

    #override window borders for negative samples - they don't get a window sum.
    window_borders_low[targets == 70] = 0
    window_borders_high[targets == 70] = 69
        
    probs = get_window_sum(outputs, window_borders_low,window_borders_high)

    binary_targets = np.ones_like(targets)
    binary_targets[targets == 70] = 0

    #now get metrics.
    auc = roc_auc_score(binary_targets, probs)

    expanded_probs = np.stack([1-probs,probs], axis =1) #plots.ROC expects probs in format [[0,1],[0,1],[0.99,0.01]] 
    roc = wandb.plots.ROC(binary_targets, expanded_probs, [0,1])

    return auc, roc


def validate(model: torch.nn.Module, valid_data: DataLoader) -> float:
    '''Run over the validation data. Average loss over the full set.
    '''
    model.eval()

    total_loss = 0
    all_targets = []
    all_preds = []

    raw_targets = []
    raw_preds = []
    for i, batch in enumerate(valid_data):
        data, targets = batch
        data = data.to(device)
        targets = targets.to(device)

        loss, prediction_probs = model(data, targets= targets)
        total_loss += loss.item()
        
        #use sentinel score as binary label for SP detection - evaluate score of not having a SP (sp positive class = 0).
        last_pos = data.shape[-1]-1
        sp_false = (targets== last_pos).detach().cpu().numpy() * 1.0
        sp_false_scores = prediction_probs[:,-1].detach().cpu().numpy()
        #change so that SP presence is true label.
        sp_true = 1- sp_false
        sp_true_scores = 1- sp_false_scores
        all_targets.append(sp_true)
        all_preds.append(sp_true_scores)
        
        #For CS metrics
        raw_preds.append(prediction_probs.detach().cpu().numpy())
        raw_targets.append(targets.detach().cpu().numpy())
        #logger.info(f'Batch {i}: Investigate manually')
        #logger.info(f'true labels: {targets[:30]}')
        #logger.info(f'max scores: {prediction_probs.detach().cpu().numpy().argmax(axis =1)[:30]}')

    #global SP detection metrics
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_pred_thresholded = (y_pred >= 0.5) *1 
    prc = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred_thresholded)

    expanded_probs = np.stack([1-y_pred,y_pred], axis =1) #plots.ROC expects probs in format [[0,1],[0,1],[0.99,0.01]] 
    det_roc_curve = wandb.plots.ROC(y_true, expanded_probs, [0,1])

    cs_precision, cs_recall = compute_cs_detection(np.concatenate(raw_preds), np.concatenate(raw_targets))
    cs_auc, cs_roc_curve = compute_cs_detection_no_threshold(np.concatenate(raw_preds), np.concatenate(raw_targets))



    val_metrics = {'loss': total_loss / len(valid_data), 'AUC detection': auc, 'AUPRC detection' : prc, 'MCC detection': mcc, 'CS Precision': cs_precision, 
                    'CS Recall': cs_recall, 'CS AUC': cs_auc}
    return total_loss / len(valid_data), val_metrics, cs_roc_curve, det_roc_curve


def main_training_loop(args: argparse.ArgumentParser):
    if args.enforce_walltime == True:
        loop_start_time = time.time()
        logger.info('Started timing loop')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #Setup Model
    #TODO how to inject new num_labels here?
    logger.info(f'Loading pretrained model in {args.resume}')
    config = ProteinAWDLSTMConfig.from_pretrained(args.resume)
    #override old model config from commandline args
    #setattr(config, 'num_labels', args.num_labels)
    setattr(config, 'classifier_hidden_size', args.classifier_hidden_size)
    model = ProteinAWDLSTMPointerSentinelModel.from_pretrained(args.resume, config = config)    
    #training logger
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    experiment_name = f"{args.experiment_name}_{model.base_model_prefix}_{time_stamp}"
    viz = visualization.get(args.output_dir, experiment_name, local_rank = -1) #debug=args.debug) #this -1 means traning is not distributed, debug makes experiment dry run for wandb

    train_data = PointerSentinelThreeLineFastaDataset(os.path.join(args.data, 'train.fasta'))
    val_data = PointerSentinelThreeLineFastaDataset(os.path.join(args.data, 'valid.fasta'))


    train_loader = DataLoader(train_data, batch_size =args.batch_size, collate_fn= train_data.collate_fn, shuffle = True)
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
        
    #keep track of best loss
    stored_loss = 100000000
    learning_rate_steps = 0

    global_step = 0
    for epoch in range(1, args.epochs+1):
        logger.info(f'Starting epoch {epoch}')
        viz.log_metrics({'Learning Rate': optimizer.param_groups[0]['lr'] }, "train", global_step)

        epoch_start_time = time.time()
        start_time = time.time() #for lr update interval
        
        for i, batch in enumerate(train_loader):

            data, targets = batch
            loss = training_step(model, data, targets, optimizer, args, i)
            viz.log_metrics({'loss': loss}, "train", global_step)
            global_step += 1

        logger.info(f'Step {global_step}, Epoch {epoch}: validating for {len(val_loader)} Validation steps')
        val_loss, val_metrics, roc_curve, det_roc_curve = validate(model, val_loader)
        #prc, auc, mcc = metrics
        #val_metrics = {'loss': val_loss, 'AUC': auc, 'AUPRC' : prc, 'MCC': mcc}
        viz.log_metrics(val_metrics, "val", global_step)
        if epoch == args.epochs:
           viz.log_metrics({'CS roc curve': roc_curve},  "val", global_step)
           viz.log_metrics({'Detection roc curve': det_roc_curve}, 'val', global_step)


        if val_loss < stored_loss:
            model.save_pretrained(args.output_dir)
            #also save with apex
            if torch.cuda.is_available():
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict()
                    }
                torch.save(checkpoint, os.path.join(args.output_dir, 'amp_checkpoint.pt'))
                logger.info(f'New best model with loss {loss}, Saving model, training step {global_step}')
            stored_loss = loss

        if  args.enforce_walltime == True and (time.time() - loop_start_time) > 84600: #23.5 hours
            logger.info('Wall time limit reached, ending training early')
            return val_loss



        logger.info(f'Epoch {epoch} training complete')
        logger.info(f'Epoch {epoch}, took {time.time() - epoch_start_time:.2f}.\t Train loss: {loss:.2f}')

    return val_loss



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='AWD-LSTM language modeling')
    parser.add_argument('--data', type=str, default='../data/data/signalp_5_data/',
                        help='location of the data corpus. Expects test, train and valid .txt')

    #args relating to training strategy.
    parser.add_argument('--lr', type=float, default=10,
                        help='initial learning rate')
    parser.add_argument('--lr_step', type = float, default = 0.9,
                        help = 'factor by which to multiply learning rate at each reduction step')
    parser.add_argument('--update_lr_steps', type = int, default = 6000,
                        help = 'After how many update steps to check for learning rate update')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=100,
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
    parser.add_argument('--resume', type=str,  default='',
                        help='path of model to resume (directory containing .bin and config.json')
    parser.add_argument('--experiment_name', type=str,  default='AWD_LSTM_LM',
                        help='experiment name for logging')
    parser.add_argument('--enforce_walltime', type=bool, default =True,
                        help='Report back current result before 24h wall time is over')


    #args for model architecture
    parser.add_argument('--classifier_hidden_size', type=int, default=128, metavar='N',
                        help='Hidden size of the classifier head MLP')
    parser.add_argument('--num_labels', type=int, default=6, metavar='N',
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