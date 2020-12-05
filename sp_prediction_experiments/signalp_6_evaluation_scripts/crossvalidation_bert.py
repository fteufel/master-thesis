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

from models.multi_crf_bert import BertSequenceTaggingCRF, ProteinBertTokenizer
from train_scripts.utils.signalp_dataset import RegionCRFDataset
from train_scripts.downstream_tasks.metrics_utils import get_metrics_multistate
from tqdm import tqdm
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default= '../data/signal_peptides/signalp_updated_data/signalp_6_train_set.fasta')
    parser.add_argument('--model_base_path', type=str, default = '/work3/felteu/tagging_checkpoints/signalp_6')
    parser.add_argument('--randomize_kingdoms', action='store_true')
    parser.add_argument('--output_file', type=str, default = 'crossval_metrics.csv')

    args = parser.parse_args()

    tokenizer=ProteinBertTokenizer.from_pretrained("/zhome/1d/8/153438/experiments/master-thesis/resources/vocab_with_kingdom", do_lower_case=False)


    # Collect results + header info to build output df
    results_list = []

    partitions = [0,1,2,3,4]

    #for result collection
    metrics_list = []
    checkpoint_list = []
    for partition in tqdm(partitions):
        # Load data
        dataset = RegionCRFDataset(args.data, tokenizer=tokenizer, partition_id = [partition], add_special_tokens=True)

        if args.randomize_kingdoms:
            import random
            kingdoms =list(set(dataset.kingdom_ids))
            random_kingdoms = random.choices(kingdoms, k=len(dataset.kingdom_ids))
            dataset.kingdom_ids = random_kingdoms
            print('randomized kingdom IDs')

        dl = torch.utils.data.DataLoader(dataset, collate_fn = dataset.collate_fn, batch_size =50)

        # Put together list of checkpoints
        checkpoints = [os.path.join(args.model_base_path, f'test_{partition}_val_{x}') for x in set(partitions).difference({partition})]

        # Run
        
        for checkpoint in checkpoints:

            model = BertSequenceTaggingCRF.from_pretrained(checkpoint)
            metrics = get_metrics_multistate(model, dl)
            metrics_list.append(metrics) #save metrics
            #checkpoint_list.append(f'test_{partition}_val_{x}') #save name of checkpoint


    df = pd.DataFrame.from_dict(metrics_list)
    #df.index = checkpoint_list

    df.T.to_csv(args.output_file)

if __name__ == '__main__':
    main()
