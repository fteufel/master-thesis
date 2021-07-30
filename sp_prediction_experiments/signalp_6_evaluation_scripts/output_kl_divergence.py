'''
Investigate the KL divergence of different models
to the full 20 models on the out of distribution test set.

Can only be done for global probs and marginal probs, not
for the viterbi path obviously.

Note that this implementation is inefficient if you want to compare a
subset of the full 20. Then you could just save the probs of the 20 and
average those as needed.
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
import numpy as np
import pandas as pd
from scipy.special import kl_div

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_data_array(model: torch.nn.Module, input_ids:  np.ndarray, input_mask: np.ndarray, batch_size = 100):
    '''Run numpy array dataset. This function takes care of batching
    and putting the outputs back together.
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_global_probs = []
    all_pos_probs = []

    total_loss = 0
    b_start = 0
    b_end = batch_size

    while b_start < len(input_ids):

        data = input_ids[b_start:b_end,:]
        data = torch.tensor(data)
        data = data.to(device)
        mask = input_mask[b_start:b_end, :]
        mask = torch.tensor(mask)
        mask = mask.to(device)


        with torch.no_grad():
            global_probs, pos_probs, pos_preds = model(data, global_targets = None, input_mask = mask)

        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_probs.append(pos_probs.detach().cpu().numpy())

        b_start = b_start + batch_size
        b_end = b_end + batch_size


    all_global_probs = np.concatenate(all_global_probs)
    all_pos_probs = np.concatenate(all_pos_probs)

    return all_global_probs, all_pos_probs



def run_data_ensemble(model: torch.nn.Module, 
                     base_path, 
                     input_ids: np.ndarray,
                     input_mask: np.ndarray,
                     do_not_average=False, 
                     partitions = [0,1,2,3,4]):


    result_list = []

    checkpoint_list = []
    name_list = []
    for outer_part in partitions:
        for inner_part in [0,1,2,3,4]:
            if inner_part != outer_part:
                path = os.path.join(base_path, f'test_{outer_part}_val_{inner_part}')
                checkpoint_list.append(path)
                name_list.append(f'T{outer_part}V{inner_part}')

 
    for path in tqdm(checkpoint_list):
        model_instance = model.from_pretrained(path)
        results = run_data_array(model_instance, input_ids, input_mask)

        result_list.append(results)


    output_obj = list(zip(*result_list)) #repacked

    return output_obj + [name_list]





def main(args):

    # Set up data
    tokenizer=ProteinBertTokenizer.from_pretrained("/zhome/1d/8/153438/experiments/master-thesis/resources/vocab_with_kingdom", do_lower_case=False)

    df = pd.read_csv(args.data, sep='\t')

    seqs = df['Sequence'].apply(lambda x: x[:70]) #truncate
    tokenized = seqs.apply(lambda x: tokenizer.encode(x, kingdom_id=args.kingdom))
    tokenized =  tokenized.apply(lambda x: x + [0] * (73-len(x))) #pad #73 = 70 +cls + kingdom + sep
    input_ids = np.vstack(tokenized)

    input_mask = (input_ids>0) *1

    model = BertSequenceTaggingCRF


    # Get outputs for first model set
    global_probs_full, pos_probs_full = run_data_ensemble(model, base_path=args.base_path, input_ids=input_ids, input_mask=input_mask)

    # Get outputs for second model set
    global_probs, pos_probs = run_data_ensemble(model, base_path=args.base_path, input_ids=input_ids, input_mask=input_mask)

    # Compute KL divergences

    div_global =  kl_div(global_probs_full, global_probs)
    