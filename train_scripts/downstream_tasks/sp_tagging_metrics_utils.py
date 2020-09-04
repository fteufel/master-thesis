import numpy as np
from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score, recall_score, precision_score
from typing import Dict, Tuple

SIGNALP_VOCAB = ['S', 'I' , 'M', 'O', 'T', 'L'] #NOTE eukarya only uses {'I', 'M', 'O', 'S'}
SIGNALP_GLOBAL_LABEL_DICT = {'NO_SP':0, 'SP':1,'LIPO':2, 'TAT':3}

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