'''
Predict the remaining sequences that were not used in the test set.
For each model, predict each sequence in the remainder.

Edgelists: files in format identifier,max_identity
Made using ggsearch36.

'''
import torch
import os
import sys
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") 
from typing import List, Tuple
import argparse

from models.signalp_5 import SignalP5Model
from train_scripts.utils.signalp_dataset import SignalP5Dataset
from train_scripts.downstream_tasks.metrics_utils import compute_metrics, run_data, tagged_seq_to_cs_multiclass
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
from itertools import permutations
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def read_edgelist(edgelist):
    '''Parse the maximum identity edges list into a dict.'''
    edges_out = {}
    with open(edgelist,'r') as f:
        for idx, edge in enumerate(f):
            seq,ident =  edge.strip().split(',')
            edges_out[seq.lstrip('>')] = float(ident)

    return edges_out

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default= '../data/signal_peptides/signalp_updated_data/signalp_6_train_set.fasta')
    parser.add_argument('--remainder_data', type=str, default= '/zhome/1d/8/153438/experiments/signalp-6.0/data/remainder.fasta')
    parser.add_argument('--edgelist_base_path', type=str, default = '/zhome/1d/8/153438/experiments/signalp-6.0/data/temp')

    parser.add_argument('--model_base_path', type=str, default = '/work3/felteu/tagging_checkpoints/signalp_5')
    parser.add_argument('--output_file', type=str, default = 'remainder_stratified_crossval_sp5.csv')

    args = parser.parse_args()


    # Load data - no need to subset.
    dataset = SignalP5Dataset(args.remainder_data)
    dl = torch.utils.data.DataLoader(dataset, collate_fn = dataset.collate_fn, batch_size =50)

    remainder_ids  = [x.split('|')[0].lstrip('>') for x in dataset.identifiers]



    #first, get a prediction for each sequence and model
    partitions = [0,1,2]
    checkpoint_dict = {}
    #iterate over train partitions
    for train_partition in tqdm(partitions, position=0, leave=True):

        # Load the identifiers of the train partition and the edgelist
        train_ids = SignalP5Dataset(args.train_data, partition_id=[train_partition]).identifiers
        train_ids = [x.split('|')[0].lstrip('>') for x in train_ids]

        max_identities = read_edgelist(os.path.join(args.edgelist_base_path, f'edgelist_train_{train_partition}.csv'))
        max_identities = np.array([max_identities[seq] for seq in remainder_ids]) #ensure same sorting, nut guaranteed from reading in.

        #get a set that contains the possible test and val partitions
        testval = set(partitions).difference({train_partition})
        #make permutations of the two
        checkpoints = [os.path.join(args.model_base_path, f'test_{x}_val_{y}') for (x,y) in permutations(testval) ]
        
        # Run and gather results in dict of dicts
        for checkpoint in checkpoints:
            checkpoint_dict[checkpoint]= {}

            model = SignalP5Model.from_pretrained(checkpoint)

            # get predictions
            all_global_targets, all_global_probs, all_targets, all_pos_preds = run_data(model, dl)
            all_cs = tagged_seq_to_cs_multiclass(all_targets, sp_tokens= [9,10,11])
            all_cs_preds = tagged_seq_to_cs_multiclass(all_pos_preds, sp_tokens= [9,10,11])

            checkpoint_dict[checkpoint]['cleavage_sites'] = all_cs_preds
            checkpoint_dict[checkpoint]['global_probs'] = all_global_probs
            checkpoint_dict[checkpoint]['identities'] = max_identities



    #now we can compute the metrics in each bin, for each model.
    metrics_list = []
    for checkpoint in checkpoint_dict.keys():
        pred_cs =  checkpoint_dict[checkpoint]['cleavage_sites']
        pred_global_label =  checkpoint_dict[checkpoint]['global_probs'].argmax(axis=1)
        identities = checkpoint_dict[checkpoint]['identities']

        all_metrics = {}
        for lower_bound in range(0,100,10):
            lower_bound = lower_bound/100

            select_idx = (identities > lower_bound) & (identities <= lower_bound+10)

            metrics = compute_metrics(all_global_targets[select_idx], pred_global_label[select_idx], 
                                        all_cs[select_idx], pred_cs[select_idx], np.array(dataset.kingdom_ids)[select_idx])

            metrics['count'] = select_idx.sum()
            metrics['multiclass_corrcoef'] = matthews_corrcoef(all_global_targets[select_idx], pred_global_label[select_idx])


            for k in metrics.keys():
                all_metrics[f'bin_{lower_bound}_'+k] = metrics[k]

        metrics_list.append(all_metrics)


    df = pd.DataFrame.from_dict(metrics_list)
    #df.index = checkpoint_list

    df.T.to_csv(args.output_file)



if __name__ == '__main__':
    main()


