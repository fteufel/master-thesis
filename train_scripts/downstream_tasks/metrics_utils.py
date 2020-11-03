'''
Utils to calculate metrics for SP tagging models.
Last version had the shortcoming of rerunning subsets of the data when getting metrics per kingdom/one-vs-all mccs.
Try to avoid that.
'''

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, recall_score, precision_score
from typing import List, Dict, Union, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIGNALP_VOCAB = ['S', 'I' , 'M', 'O', 'T', 'L'] #NOTE eukarya only uses {'I', 'M', 'O', 'S'}
SIGNALP_GLOBAL_LABEL_DICT = {'NO_SP':0, 'SP':1,'LIPO':2, 'TAT':3}
REVERSE_GLOBAL_LABEL_DICT = {0 : 'NO_SP', 1: 'SP', 2: 'LIPO', 3: 'TAT'}

def run_data(model, dataloader):
    '''run all the data of a DataLoader, concatenate and return outputs'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_targets = []
    all_global_targets = []
    all_global_probs = []
    all_pos_preds = []

    total_loss = 0
    for i, batch in enumerate(dataloader):

        if hasattr(dataloader.dataset, 'kingdom_ids') and hasattr(dataloader.dataset, 'sample_weights'):
            data, targets, input_mask, global_targets, sample_weights, kingdom_ids = batch
            kingdom_ids = kingdom_ids.to(device)
            sample_weights = sample_weights.to(device)
        elif  hasattr(dataloader.dataset, 'sample_weights') and not hasattr(dataloader.dataset, 'kingdom_ids'):
            data, targets, input_mask, global_targets, sample_weights = batch
            kingdom_ids = None
        else:
            data, targets, input_mask, global_targets, sample_weights = batch
            kingdom_ids = None
            
        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)
        global_targets = global_targets.to(device)
        with torch.no_grad():
            loss, global_probs, pos_probs, pos_preds = model(data, global_targets = global_targets, targets=  targets, input_mask = input_mask, kingdom_ids=kingdom_ids)

        total_loss += loss.item()
        all_targets.append(targets.detach().cpu().numpy())
        all_global_targets.append(global_targets.detach().cpu().numpy())
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)

    return all_global_targets, all_global_probs, all_targets, all_pos_preds

def run_data_regioncrfdataset(model, dataloader):
    '''run all the data of a DataLoader, concatenate and return outputs'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_targets = []
    all_global_targets = []
    all_global_probs = []
    all_pos_preds = []
    all_cs = []

    total_loss = 0
    for i, batch in enumerate(dataloader):

        if hasattr(dataloader.dataset, 'kingdom_ids') and hasattr(dataloader.dataset, 'sample_weights'):
            data, targets, input_mask, global_targets, cleavage_sites, sample_weights, kingdom_ids = batch
            kingdom_ids = kingdom_ids.to(device)
            sample_weights = sample_weights.to(device)
        elif  hasattr(dataloader.dataset, 'sample_weights') and not hasattr(dataloader.dataset, 'kingdom_ids'):
            data, targets, input_mask, global_targets, cleavage_sites, sample_weights = batch
            kingdom_ids = None
        else:
            data, targets, input_mask, global_targets,cleavage_sites, sample_weights = batch
            kingdom_ids = None
            
        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)
        global_targets = global_targets.to(device)
        with torch.no_grad():
            global_probs, pos_probs, pos_preds = model(data, input_mask = input_mask, kingdom_ids=kingdom_ids)

        all_targets.append(targets.detach().cpu().numpy())
        all_global_targets.append(global_targets.detach().cpu().numpy())
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())
        all_cs.append(cleavage_sites)

    import ipdb; ipdb.set_trace()

    all_targets = np.concatenate(all_targets)
    all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)
    all_cs = np.concatenate(all_cs)

    return all_global_targets, all_global_probs, all_targets, all_pos_preds, all_cs

def run_data_array(model, sequence_data_array, batch_size = 40):
    '''run all the data of a np.array, concatenate and return outputs
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_global_probs = []
    all_pos_preds = []

    total_loss = 0
    b_start = 0
    b_end = batch_size
    while b_start < len(sequence_data_array):

        data = sequence_data_array[b_start:b_end,:]
        data = torch.tensor(data)
        data = data.to(device)
        #global_targets = global_targets.to(device)
        input_mask = None
        with torch.no_grad():
            global_probs, pos_probs, pos_preds = model(data, input_mask = torch.ones_like(data))

       # total_loss += loss.item()
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())

        b_start = b_start + batch_size
        b_end = b_end + batch_size

    #all_targets = np.concatenate(all_targets)
    #all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)

    return all_global_probs, all_pos_preds

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

def find_cs_tag(tagged_seqs: np.ndarray, cs_tokens = [4, 9, 14]):
    '''Convert a sequences of tokens to the index of the cleavage site.
    Inputs:
        tagged_seqs: (batch_size, seq_len) integer array of position-wise labels , with "C" tags for cleavage sites
        cs_tokens = label tokens that indicate a cleavage site
    Returns:
        cs_sites: (batch_size) integer array of position that is a CS. -1 if no SP present in sequence.
    #NOTE this is still the same as tagged_seq_to_cs_multiclass. Implement unique-checks later (only one CS per seq)
    '''
    def get_last_sp_idx(x: np.ndarray) -> int:
        '''Func1d to get the last index that is tagged as SP. use with np.apply_along_axis. '''
        sp_idx = np.where(np.isin(x, cs_tokens))[0]
        if len(sp_idx)>0:
            #TODO remove or rework to warning, in training might well be that continuity is not learnt yet (i don't enforce it in the crf setup)
            #assert sp_idx.max() +1 - sp_idx.min() == len(sp_idx) #assert continuity. CRF should actually guarantee that (if learned correcty)
            max_idx = sp_idx.max()+1
        else:
            max_idx = -1
        return max_idx

    cs_sites = np.apply_along_axis(get_last_sp_idx, 1, tagged_seqs)
    return cs_sites    

def compute_metrics(all_global_targets: np.ndarray, all_global_preds: np.ndarray, all_cs_targets: np.ndarray, all_cs_preds: np.ndarray, 
                    all_kingdom_ids: List[str]): 
    '''Compute the metrics used in the SignalP 5.0 supplement.
    Returns all metrics as a dict. 
    '''    
    metrics_dict = {}
    #get simple overall global metrics
    metrics_dict['mcc'] = matthews_corrcoef(all_global_targets, all_global_preds)

    #subset data for each kingdom
    for kingdom in np.unique(all_kingdom_ids):
        kingdom_targets = all_global_targets[all_kingdom_ids == kingdom] 
        kingdom_preds = all_global_preds[all_kingdom_ids == kingdom]
        kingdom_cs_targets = all_cs_targets[all_kingdom_ids == kingdom]
        kingdom_cs_preds = all_cs_preds[all_kingdom_ids == kingdom]

        #subset data for each sp type
        for sp_type in np.unique(kingdom_targets):
            #one-vs-all mcc
            targets_mcc2 = kingdom_targets.copy()
            preds_mcc2 = kingdom_preds.copy()
            #make same label for all that are not target type
            targets_mcc2[targets_mcc2 != sp_type ] = -1
            preds_mcc2[preds_mcc2 !=sp_type] = -1
            metrics_dict[f'{kingdom}_{REVERSE_GLOBAL_LABEL_DICT[sp_type]}_mcc2'] = matthews_corrcoef(targets_mcc2, preds_mcc2)
            #one-vs-no_sp mcc
            targets_mcc1 = kingdom_targets[np.isin(kingdom_targets, [sp_type, 0])] #TODO hardcode or replace
            preds_mcc1 = kingdom_preds[np.isin(kingdom_targets, [sp_type, 0])]
            metrics_dict[f'{kingdom}_{REVERSE_GLOBAL_LABEL_DICT[sp_type]}_mcc1'] = matthews_corrcoef(targets_mcc1, preds_mcc1)

            #subset current type + type 0 (no sp)
            true_CS, pred_CS = kingdom_cs_targets[np.isin(kingdom_targets, [sp_type, 0])], kingdom_cs_preds[np.isin(kingdom_targets, [sp_type, 0])]

            #get CS metrics
            #For the precision calculation, you have a vector with the position of the true and predicted cs. 
            # So if the model doesn’t predict a SP, you would have -1 in the pred_CS, I assume. 
            # The first thing that you want to do is subset proteins predicted as having a SP (pred_CS != -1) and also having a true SP
            # so we can compare the cleavage sites (true_CS != -1), which I don’t see in your code. 
            # What you want to have is true_positive = true_CS[(true_CS != -1) & (pred_CS != -1)], 
            # and pred_positive = pred_CS[(true_CS != -1) & (pred_CS != -1)], 
            # The precision would be then: (pred_positive == true_positive).sum()/(pred_CS != -1).sum(). 
            # Note that we divide by the total number of proteins predicted as having an SP. 
            # Let me know if I need to explain this better.
            
            # The first thing that you want to do is subset proteins predicted as having a SP (pred_CS != -1) and also having a true SP
            true_positive = true_CS[(true_CS != -1) & (pred_CS != -1)]
            pred_positive =pred_CS[(true_CS != -1) & (pred_CS != -1)]

            for window_size in [0,1,2,3]:
                correct_in_window = pred_positive == true_positive #window 0
                correct_in_window = (pred_positive >= true_positive-window_size) & (pred_positive <= true_positive+window_size)
                precision = correct_in_window.sum()/(pred_CS != -1).sum()
                recall = correct_in_window.sum()/(true_CS != -1).sum()
                
                metrics_dict[f'{kingdom}_{REVERSE_GLOBAL_LABEL_DICT[sp_type]}_recall_window_{window_size}'] = recall
                metrics_dict[f'{kingdom}_{REVERSE_GLOBAL_LABEL_DICT[sp_type]}_precision_window_{window_size}'] = precision

            use_legacy=False
            if use_legacy:
                #old implementation 
                

                positive_samples_targets = true_CS[true_CS != -1]
                positive_samples_preds = pred_CS[true_CS  != -1]
                negative_samples_preds = pred_CS[true_CS == -1] #no need for targets here, all -1
                assert len(negative_samples_preds) + len(positive_samples_preds) == len(pred_CS)
                #false negatives: CS that were not found (pred_CS = -1) and CS that are out of window
                # absolute = independent of window choice
                absolute_false_negatives = (positive_samples_preds ==  -1).sum()
                #false positives: CS that should not exist (true_CS = -1)
                absolute_false_positives = (negative_samples_preds !=-1).sum()
                for window_size in [0,1,2,3]:
                    window_true_pos = (positive_samples_preds >= positive_samples_targets-window_size) & (positive_samples_preds <= positive_samples_targets+window_size)
                    window_true_pos = window_true_pos.sum()
                    
                    # Recall = TP / TP+FN
                    metrics_dict[f'{kingdom}_{REVERSE_GLOBAL_LABEL_DICT[sp_type]}_recall_window_{window_size}'] = window_true_pos / len(positive_samples_targets)
                    #Precision = TP/ TP+FP
                    metrics_dict[f'{kingdom}_{REVERSE_GLOBAL_LABEL_DICT[sp_type]}_precision_window_{window_size}'] = window_true_pos / (window_true_pos + absolute_false_positives)
    return metrics_dict



def get_metrics(model, data = Union[Tuple[np.ndarray, np.ndarray, np.ndarray], torch.utils.data.DataLoader], cs_tagged = False):
    
    sp_tokens = [3,7,11] if model.use_large_crf else [0,4,5]
    print(f'Using SP tokens {sp_tokens} to infer cleavage site.')
 
    #handle array datasets (small organism test sets)
    if type(data) == tuple:
        data, global_targets, all_cs_targets = data
        all_global_probs, all_pos_preds = run_data_array(model, data)
        kingdom_ids = ['DEFAULT'] * len(global_targets) #do not split by kingdoms, make dummy kingdom
    #handle dataloader (signalp data)
    else:
        all_global_targets, all_global_probs, all_targets, all_pos_preds = run_data(model, data)

        if cs_tagged:
            all_cs_targets = find_cs_tag(all_targets)
        else:
            all_cs_targets = tagged_seq_to_cs_multiclass(all_targets, sp_tokens= sp_tokens)
            
        #parse kingdom ids
        parsed = [ element.lstrip('>').split('|') for element in data.dataset.identifiers]
        acc_ids, kingdom_ids, type_ids, partition_ids = [np.array(x) for x in list(zip(*parsed))]

    all_global_preds = all_global_probs.argmax(axis =1)
    if cs_tagged:
        all_cs_preds = find_cs_tag(all_pos_preds)
    else:
        all_cs_preds = tagged_seq_to_cs_multiclass(all_pos_preds, sp_tokens= sp_tokens)
    metrics = compute_metrics(all_global_targets, all_global_preds, all_cs_targets, all_cs_preds, all_kingdom_ids=kingdom_ids)

    return metrics


def get_metrics_multistate(model, data = Union[Tuple[np.ndarray, np.ndarray, np.ndarray], torch.utils.data.DataLoader]):
    '''Works with multi-state CRF and corresponding RegionDataset'''
    
    sp_tokens = [5, 11, 19]
    print(f'Using SP tokens {sp_tokens} to infer cleavage site.')
 
    #handle array datasets (small organism test sets)
    if type(data) == tuple:
        data, all_global_targets, all_cs_targets = data
        all_global_probs, all_pos_preds = run_data_array(model, data)
        kingdom_ids = ['DEFAULT'] * len(all_global_targets) #do not split by kingdoms, make dummy kingdom for compatibility with same functions
    #handle dataloader (signalp data)
    else:
        all_global_targets, all_global_probs, all_targets, all_pos_preds, all_cs = run_data_regioncrfdataset(model, data)
        all_cs_targets = all_cs
            
        #parse kingdom ids
        parsed = [ element.lstrip('>').split('|') for element in data.dataset.identifiers]
        acc_ids, kingdom_ids, type_ids, partition_ids = [np.array(x) for x in list(zip(*parsed))]

    all_global_preds = all_global_probs.argmax(axis =1)
    print(all_global_preds)

    all_cs_preds = tagged_seq_to_cs_multiclass(all_pos_preds, sp_tokens= sp_tokens)
    metrics = compute_metrics(all_global_targets, all_global_preds, all_cs_targets, all_cs_preds, all_kingdom_ids=kingdom_ids)

    return metrics

