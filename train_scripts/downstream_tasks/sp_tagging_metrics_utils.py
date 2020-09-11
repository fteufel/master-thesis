import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score, recall_score, precision_score
from typing import Dict, Tuple
import pandas as pd
import sys
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet

from models.sp_tagging_prottrans import XLNetSequenceTaggingCRF, ProteinXLNetTokenizer
from train_scripts.utils.signalp_dataset import PartitionThreeLineFastaDataset

SIGNALP_VOCAB = ['S', 'I' , 'M', 'O', 'T', 'L'] #NOTE eukarya only uses {'I', 'M', 'O', 'S'}
SIGNALP_GLOBAL_LABEL_DICT = {'NO_SP':0, 'SP':1,'LIPO':2, 'TAT':3}

def tagged_seq_to_cs_multiclass(tagged_seqs: np.ndarray, sp_tokens = [0,4,5]):
    '''Convert a sequences of tokens to the index of the cleavage site.
    Inputs:
        tagged_seqs: (batch_size, seq_len) integer array of position-wise labels
        sp_tokens: label tokens that indicate a signal peptide
    Returns:
        cs_sites: (batch_size) integer array of last position that is a SP. -1 if no SP present in sequence.
    '''
    def get_last_sp_idx(x: np.ndarray) -> int:
        '''Func1d to get the last index that is tagged as SP. use with np.apply_along_axis. '''
        sp_idx = np.where(np.isin(x, sp_tokens))[0]
        if len(sp_idx)>0:
            #TODO remove or rework to warning, in training might well be that continuity is not learnt yet (i don't enforce it in the crf setup)
            #assert sp_idx.max() +1 - sp_idx.min() == len(sp_idx) #assert continuity. CRF should actually guarantee that (if learned correcty)
            max_idx = sp_idx.max()+1
        else:
            max_idx = -1
        return max_idx

    cs_sites = np.apply_along_axis(get_last_sp_idx, 1, tagged_seqs)
    return cs_sites


def report_metrics(true_global_labels: np.ndarray, pred_global_labels: np.ndarray, true_sequence_labels: np.ndarray, 
                    pred_sequence_labels: np.ndarray) -> Dict[str, float]:
    '''Utility function to get metrics from model output'''
    true_cs = tagged_seq_to_cs_multiclass(true_sequence_labels)
    pred_cs = tagged_seq_to_cs_multiclass(pred_sequence_labels)
    #TODO decide on how to calcuate cs metrics: ignore no cs seqs, or non-detection implies correct cs?
    pred_cs = pred_cs[~(true_cs == -1)]
    true_cs = true_cs[~(true_cs == -1)]
    true_cs[(true_cs == -1)] = -1
    pred_cs[(true_cs == -1)] = -1

    #applying a threhold of 0.25 (SignalP) to a 4 class case is equivalent to the argmax.
    pred_global_labels_thresholded = pred_global_labels.argmax(axis=1)
    metrics_dict = {}
    metrics_dict['CS Recall'] = recall_score(true_cs, pred_cs, average='micro')
    metrics_dict['CS Precision'] = precision_score(true_cs, pred_cs, average='micro')
    metrics_dict['CS MCC'] = matthews_corrcoef(true_cs, pred_cs)
    metrics_dict['Detection MCC'] = matthews_corrcoef(true_global_labels, pred_global_labels_thresholded)

    return metrics_dict

def get_discrepancy_rate(pred_global_labels: np.ndarray, pred_sequence_labels: np.ndarray, sp_tokens = [0,4,5]) -> Tuple[float, float]:
    '''Get ratio of sequences that got a global SP label, but have none tagged in their sequence.  
    Right now, only reports total. Adapt output to report per-class ratios.  
    In a poorly trained model, these ratios can be bigger than 1.

    Inputs:  
        `pred_global_labels` :     (batch_size x seq_len x n_labels) global label probabilities
        `pred_sequence_labels` :   (batch_size x seq_len) integer array of position-wise labels

    Outputs:   
        `disc_ratio` :              ratio of sequences that have a discrepancy.
        `multi_ratio` :             ratio of sequences that have more than one SP type tagged
    '''
    pred_global_labels_thresholded = pred_global_labels.argmax(axis=1)

    #convert sequence label to global label -> check if they have any SP tag.
    seq_sp  = np.any(pred_sequence_labels == 0, axis =1) #SPI
    seq_tat = np.any(pred_sequence_labels == 4, axis =1) #TAT
    seq_lip = np.any(pred_sequence_labels == 5, axis =1) #LIPO

    glob_sp = pred_global_labels_thresholded == 1 
    glob_tat= pred_global_labels_thresholded == 3  
    glob_lip= pred_global_labels_thresholded == 2 

    sp_discrepancy  = (glob_sp != seq_sp)
    tat_discrepancy = (glob_tat != seq_tat)
    lip_discrepancy = (glob_lip != seq_lip)

    total_discrepancy = sp_discrepancy.sum() + tat_discrepancy.sum() + lip_discrepancy.sum()
    total_global_tagged = glob_sp.sum() + glob_tat.sum() + glob_lip.sum()

    #check if sequences have mixed tagging (no two types of SPs can be present at the same time)
    mixed_tagged = seq_sp == seq_tat
    mixed_tagged = mixed_tagged = seq_lip
    mixed_tagged = mixed_tagged.sum()

    return total_discrepancy/total_global_tagged, mixed_tagged/total_global_tagged

def get_window_accuracies_precisions(true_sequences, predicted_sequences):
    ''''Compute CS Recall and Precision for all tolerance windows.'''

    cs_pos_true = tagged_seq_to_cs_multiclass(true_sequences) #true negatives are nan
    cs_pos_pred = tagged_seq_to_cs_multiclass(predicted_sequences) #true and false negatives are nan
    #TODO decide on how to calcuate cs metrics: ignore no cs seqs, or non-detection implies correct cs?

    cs_pos_recovered = cs_pos_pred[~((cs_pos_true == -1) | (cs_pos_pred == -1))] #drop all that are predicted false or are false -> true pos (no need to apply window to false negatives, doesnt change anything)
    true_cs_positions = cs_pos_true[~((cs_pos_true == -1) | (cs_pos_pred == -1))]

    #on these cs, can use windows.
    window_0_true_pos = cs_pos_recovered == true_cs_positions
    # numpy: (c > 2) & (c < 5)
    window_1_true_pos = (cs_pos_recovered >= true_cs_positions-1) & (cs_pos_recovered <= true_cs_positions +1)
    window_2_true_pos = (cs_pos_recovered >= true_cs_positions-2) & (cs_pos_recovered <= true_cs_positions +2)
    window_3_true_pos = (cs_pos_recovered >= true_cs_positions-3) & (cs_pos_recovered <= true_cs_positions +3)

    metrics_dict = {}

    metrics_dict['accuracy_0'] = window_0_true_pos.sum() / len(cs_pos_true[~(cs_pos_true == -1)])#divide by all sequences that have a cs
    metrics_dict['accuracy_1'] = window_1_true_pos.sum() / len(cs_pos_true[~(cs_pos_true == -1)])
    metrics_dict['accuracy_2'] = window_2_true_pos.sum() / len(cs_pos_true[~(cs_pos_true == -1)])
    metrics_dict['accuracy_3'] = window_3_true_pos.sum() / len(cs_pos_true[~(cs_pos_true == -1)])

    #TODO - do I need to apply the window to the denominator too?
    #
    metrics_dict['precision_0'] = window_0_true_pos.sum() / len(cs_pos_true[~(cs_pos_pred == -1)])
    metrics_dict['precision_1'] = window_1_true_pos.sum() / len(cs_pos_true[~(cs_pos_pred == -1)])
    metrics_dict['precision_2'] = window_2_true_pos.sum() / len(cs_pos_true[~(cs_pos_pred == -1)])
    metrics_dict['precision_3'] = window_3_true_pos.sum() / len(cs_pos_true[~(cs_pos_pred == -1)])

    if metrics_dict['precision_1'] < metrics_dict['precision_0']:
        import ipdb; ipdb.set_trace()

    return metrics_dict


def get_one_vs_all_mcc(true_global_labels, pred_global_labels):
    '''Compute binary one-vs-all MCC (MCC2 in SignalP SI)'''

    labels_found = np.unique(true_global_labels)
    metrics_dict = {}
    for label in labels_found:
        #binarize
        true_override =  true_global_labels.copy()
        pred_override =  pred_global_labels.copy()
        true_override[true_override != label] = -1
        pred_override[pred_override != label] = -1
        assert( len(np.unique(pred_override)) ==2 )
        assert( len(np.unique(true_override)) ==2 )
        mcc = matthews_corrcoef(true_override, pred_override)
        metrics_dict[f' MCC2 {list(SIGNALP_GLOBAL_LABEL_DICT.keys())[label]}'] = mcc
    return metrics_dict

import torch
def validate(model: torch.nn.Module, valid_data: torch.utils.data.DataLoader) -> float:
    '''Run over the validation data. Average loss over the full set.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    all_targets = []
    all_global_targets = []
    all_global_probs = []
    all_pos_preds = []

    total_loss = 0
    for i, batch in enumerate(valid_data):
        data, targets, global_targets = batch
        data = data.to(device)
        targets = targets.to(device)
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
    window_metrics = get_window_accuracies_precisions(all_targets, all_pos_preds)
    val_metrics = {'loss': total_loss / len(valid_data), **metrics, **window_metrics }
    return (total_loss / len(valid_data)), val_metrics


def validate_kingdom_type_level(model, data_path, partition_id = [0]):
    ''''Get validation metrics on subsets of the dataset.'''
    results_list = []
    index_list = []
    tokenizer = ProteinXLNetTokenizer.from_pretrained('Rostlab/prot_xlnet', do_lower_case = False)

    ds = PartitionThreeLineFastaDataset(data_path, tokenizer, partition_id = partition_id,kingdom_id = ['EUKARYA'], type_id = ['SP', 'NO_SP'], add_special_tokens = True)
    dl = torch.utils.data.DataLoader(ds, collate_fn = ds.collate_fn, batch_size = 40)

    val_loss, val_metrics = validate(model, dl)
    results_list.append(pd.Series(val_metrics))
    index_list.append('eukarya SPI')


    for kingdom in ['ARCHAEA', 'POSITIVE', 'NEGATIVE']:
        #SPI
        ds = PartitionThreeLineFastaDataset(data_path, tokenizer, partition_id = partition_id, kingdom_id = [kingdom], type_id = ['SP', 'NO_SP'], add_special_tokens = True)
        dl = torch.utils.data.DataLoader(ds, collate_fn = ds.collate_fn, batch_size = 40)
        val_loss, val_metrics = validate(model, dl)
        results_list.append(pd.Series(val_metrics))
        index_list.append(f'{kingdom.lower()} SPI')
        #SPII
        ds = PartitionThreeLineFastaDataset(data_path, tokenizer, partition_id = partition_id, kingdom_id = [kingdom], type_id = ['LIPO', 'NO_SP'], add_special_tokens = True)
        dl = torch.utils.data.DataLoader(ds, collate_fn = ds.collate_fn, batch_size = 40)
        val_loss, val_metrics = validate(model, dl)
        results_list.append(pd.Series(val_metrics))
        index_list.append(f'{kingdom.lower()} SPII')
        #TAT
        ds = PartitionThreeLineFastaDataset(data_path, tokenizer, partition_id = partition_id, kingdom_id = [kingdom], type_id = ['TAT', 'NO_SP'], add_special_tokens = True)
        dl = torch.utils.data.DataLoader(ds, collate_fn = ds.collate_fn, batch_size = 40)
        val_loss, val_metrics = validate(model, dl)
        results_list.append(pd.Series(val_metrics))
        index_list.append(f'{kingdom.lower()} TAT')

    return pd.DataFrame(results_list, index = index_list)



def validate_mcc2(model: torch.nn.Module, valid_data: torch.utils.data.DataLoader) -> float:
    '''Run over the validation data. Average loss over the full set.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    all_targets = []
    all_global_targets = []
    all_global_probs = []
    all_pos_preds = []

    total_loss = 0
    for i, batch in enumerate(valid_data):
        data, targets, global_targets = batch
        data = data.to(device)
        targets = targets.to(device)
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

    metrics = get_one_vs_all_mcc(all_global_targets, all_global_probs.argmax(axis =1))
    val_metrics = {'loss': total_loss / len(valid_data), **metrics }
    return (total_loss / len(valid_data)), val_metrics