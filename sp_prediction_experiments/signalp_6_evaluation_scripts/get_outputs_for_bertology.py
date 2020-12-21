'''
Get attention heads, predictions and hidden states and pickle.
EDA will be done in a notebook, but model needs to run on HPC.

Just do this for one checkpoint. For more it would be tricky,
as it's not guaranteed that the same attention head would learn
the same features.
'''

import torch
import os
import sys
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") 
from typing import List, Tuple
import argparse

from transformers import BertModel
from models.multi_crf_bert import BertSequenceTaggingCRF, ProteinBertTokenizer
from train_scripts.utils.signalp_dataset import RegionCRFDataset
from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_attentions(model, dataloader):
        
    model.to(device)
    model.eval()

    attentions_batched = []
    hidden_states_batched = []
    probs_batched = []
        
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            
        data, targets, input_mask, global_targets, weights,kingdoms, cleavage_sites= batch
        input_ids = data.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            
            hidden_state, _, attentions = model(input_ids, input_mask, output_attentions=True)
            attentions_batched.append([attn.cpu() for attn in attentions])
            hidden_states_batched.append(hidden_state.cpu())

    # attentions_batched is a list of size n_layers lists, need to cat the elements at each position to
    # get the attention heads for all samples as single tensors
    attentions = [torch.cat(t) for t in zip(*attentions_batched)] 
    # List (n_batch) of Lists (n_layers) containing (b_size,..) tensors -> List (n_layers) of (n_all_samples, ...) tensors
    hidden_states =  torch.cat(hidden_states_batched)

    return attentions, hidden_states


def get_model_outputs(model, dataloader):

    model.to(device)
    model.eval()

    pos_probs_batched = []
    pos_preds_batched = []
    global_probs_batched = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        
        data, targets, input_mask, global_targets, weights,kingdoms, cleavage_sites= batch
        data = data.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            global_probs, pos_probs, pos_preds, emissions, input_mask = model(data, input_mask=input_mask, return_emissions=True)
            pos_probs_batched.append(pos_probs.cpu())
            pos_preds_batched.append(pos_preds.cpu())
            global_probs_batched.append(global_probs.cpu())


    pos_probs = torch.cat(pos_probs_batched)
    pos_preds = torch.cat(pos_preds_batched)
    global_probs = torch.cat(global_probs_batched)

    return pos_probs, pos_preds, global_probs

def main(args):

    tokenizer =ProteinBertTokenizer.from_pretrained('../resources/vocab_with_kingdom', do_lower_case=False)
    ds = RegionCRFDataset(args.data,tokenizer=tokenizer,add_special_tokens=True, partition_id=[args.partition])
    dl = torch.utils.data.DataLoader(ds, collate_fn = ds.collate_fn, batch_size =100)

    print('Saving attention heads')
    model = BertModel.from_pretrained(args.bert_checkpoint)
    attentions, hidden_states = get_attentions(model, dl)

    print('Saving model prediction outputs')
    model = BertSequenceTaggingCRF.from_pretrained(args.bert_checkpoint)
    pos_probs, pos_preds, global_probs = get_model_outputs(model, dl)

    #pickle everything

    outdict = {'attn_heads': [attns.numpy() for attns in attentions],
                'hidden_states': hidden_states.numpy(),
                'pos_probs': pos_probs.numpy(),
                'pos_preds': pos_preds.numpy(),
                'global_probs': global_probs.numpy(),
                'global_label': ds.global_labels,
                'kingdoms': ds.kingdom_ids,
                'identifiers': ds.identifiers,
                'labels': ds.labels,
                'sequences': ds.sequences,
                }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(outdict, f)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default= '../data/signal_peptides/signalp_updated_data/signalp_6_train_set.fasta')
    parser.add_argument('--bert_checkpoint', type=str, default = '/work3/felteu/tagging_checkpoints/signalp_6/test_0_val_1')
    parser.add_argument('--output_file', type=str, default = 'bert_outputs.pkl')
    parser.add_argument('--partition', type=float, default = 0)
    args = parser.parse_args()


    main(args)


