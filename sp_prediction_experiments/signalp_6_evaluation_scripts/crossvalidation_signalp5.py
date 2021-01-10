'''
Compute cross-validated metrics. Run each model
on its test set, compute all metrics and save as .csv
We save all metrics per model in the .csv, mean/sd
calculation we do later in a notebook.
'''
import torch
import os
import sys
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") 
from typing import List, Tuple
import argparse

from models.signalp_5 import SignalP5Model
from train_scripts.utils.signalp_dataset import SignalP5Dataset
from train_scripts.downstream_tasks.metrics_utils import get_metrics, get_metrics_multistate
from tqdm import tqdm
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default= '../data/signal_peptides/signalp_updated_data/signalp_6_train_set.fasta')
    parser.add_argument('--model_base_path', type=str, default = '/work3/felteu/tagging_checkpoints/signalp_5')
    parser.add_argument('--randomize_kingdoms', action='store_true')
    parser.add_argument('--output_file', type=str, default = 'crossval_metrics.csv')
    parser.add_argument('--n_partitions', type=int, default=5, help='Number of partitions, for loading the checkpoints and datasets.')
    parser.add_argument('--multistate', action='store_true', help='Model to evaluate is a multistate model, use different labels to find CS')


    args = parser.parse_args()


    # Collect results + header info to build output df
    results_list = []

    partitions = list(range(args.n_partitions))

    #for result collection
    metrics_list = []
    checkpoint_list = []
    for partition in tqdm(partitions):
        # Load data
        dataset = SignalP5Dataset(args.data, partition_id = [partition], return_region_labels=args.multistate)

        if args.randomize_kingdoms:
            import random
            kingdoms =list(set(dataset.kingdom_ids))
            random_kingdoms = random.choices(kingdoms, k=len(dataset.kingdom_ids))
            dataset.kingdom_ids = random_kingdoms
            print('randomized kingdom IDs')

        dl = torch.utils.data.DataLoader(dataset, collate_fn = dataset.collate_fn, batch_size =100)

        # Put together list of checkpoints
        checkpoints = [os.path.join(args.model_base_path, f'test_{partition}_val_{x}') for x in set(partitions).difference({partition})]

        # Run
        
        for checkpoint in checkpoints:

            model = SignalP5Model.from_pretrained(checkpoint)
            if args.multistate:
                metrics = get_metrics_multistate(model, dl, sp_tokens= [5,8,13,17,19], compute_region_metrics=False) #this relies on hardcoded stuff, not adapted
            else:
                metrics = get_metrics(model, dl, sp_tokens= [9,10,11])
            metrics_list.append(metrics) #save metrics
            #checkpoint_list.append(f'test_{partition}_val_{x}') #save name of checkpoint


    df = pd.DataFrame.from_dict(metrics_list)
    #df.index = checkpoint_list

    df.T.to_csv(args.output_file)

if __name__ == '__main__':
    main()