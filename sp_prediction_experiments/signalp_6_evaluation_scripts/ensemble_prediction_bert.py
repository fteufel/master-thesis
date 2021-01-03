'''
Predict sequences from a .tsv file with all 20 models and average.
Also averages viterbi path correctly.
TODO need to merge viterbi+emission functions to reduce overhead from model loading
Also load transition matrices during emission computation, don't reload model later.
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_averaged_emissions_from_arrays(model_checkpoint_list: List[str],  input_ids:  np.ndarray, 
                                 input_mask: np.ndarray, batch_size = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Run sequences through all models in list.
    Return only the emissions (Bert+linear projection) and average those.
    Also returns input masks, as CRF needs them when decoding.
    '''

    emission_list = []
    global_probs_list = []
    for checkpoint in tqdm(model_checkpoint_list):

        # Load model and get emissions
        model = BertSequenceTaggingCRF.from_pretrained(checkpoint)
        model.to(device)
        model.eval()

        emissions_batched = []
        masks_batched = []
        probs_batched = []

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
                global_probs, pos_probs, pos_preds, emissions, mask = model(data, input_mask=mask, return_emissions=True)
                emissions_batched.append(emissions)
                masks_batched.append(mask.cpu())
                probs_batched.append(global_probs.cpu())
            
            b_start = b_start + batch_size
            b_end = b_end + batch_size

        #covert to CPU tensors after forward pass. Save memory.
        model_emissions = torch.cat(emissions_batched).detach().cpu()
        emission_list.append(model_emissions)
        masks =  torch.cat(masks_batched) # gathering mask in inner loop is sufficent, same every time.
        model_probs = torch.cat(probs_batched).detach().cpu()
        global_probs_list.append(model_probs)

    # Average the emissions

    emissions = torch.stack(emission_list)
    probs = torch.stack(global_probs_list)
    return emissions.mean(dim=0), masks, probs.mean(dim=0)



def run_averaged_crf(model_checkpoint_list: List[str], emissions: torch.Tensor, input_mask: torch.tensor):
    '''Average weights of a list of model checkpoints,
    then run viterbi decoding on provided emissions.
    Running this on CPU should be fine, viterbi decoding
    does not use a lot of multiplications. Only one for each timestep.
    '''

    # Gather the CRF weights
    start_transitions = [] 
    transitions = []
    end_transitions = []

    for checkpoint in model_checkpoint_list:
        model = BertSequenceTaggingCRF.from_pretrained(checkpoint)
        start_transitions.append(model.crf.start_transitions.data)
        transitions.append(model.crf.transitions.data)
        end_transitions.append(model.crf.end_transitions.data)


    # Average and set model weights
    start_transitions = torch.stack(start_transitions).mean(dim=0)
    transitions = torch.stack(transitions).mean(dim=0)
    end_transitions = torch.stack(end_transitions).mean(dim=0)

    model.crf.start_transitions.data = start_transitions
    model.crf.transitions.data =  transitions
    model.crf.end_transitions.data = end_transitions

    # Get viterbi decodings
    with torch.no_grad():
        viterbi_paths = model.crf.decode(emissions=emissions, mask=input_mask.byte())

    return viterbi_paths


def main():

    parser = argparse.ArgumentParser('Get ensemble predictions')
    parser.add_argument('--data', type=str)
    parser.add_argument('--base_path', type=str, default='/work3/felteu/tagging_checkpoints/signalp_6/')
    parser.add_argument('--output_file', type=str,default='ensemble_predictions.csv')
    parser.add_argument('--kingdom', type=str, default='EUKARYA', const='EUKARYA', nargs='?', choices=['EUKARYA', 'ARCHAEA','POSITIVE', 'NEGATIVE'] )
    parser.add_argument('--n_partitions', type=int, default=5, help='Number of partitions, for loading the checkpoints and datasets.')

    args = parser.parse_args()

    args = parser.parse_args()

    tokenizer=ProteinBertTokenizer.from_pretrained("/zhome/1d/8/153438/experiments/master-thesis/resources/vocab_with_kingdom", do_lower_case=False)

    df = pd.read_csv(args.data, sep='\t')

    seqs = df['Sequence'].apply(lambda x: x[:70]) #truncate
    tokenized = seqs.apply(lambda x: tokenizer.encode(x, kingdom_id=args.kingdom))
    tokenized =  tokenized.apply(lambda x: x + [0] * (73-len(x))) #pad #73 = 70 +cls + kingdom + sep
    input_ids = np.vstack(tokenized)

    input_mask = (input_ids>0) *1


    #checkpoints = [os.path.join(args.model_base_path, f'test_{partition}_val_{x}') for x in set(partitions).difference({partition})]
    checkpoint_list = []
    for part1 in range(args.n_partitions):
        for part2 in range(args.n_partitions):
            if part1 != part2:
                checkpoint_list.append(os.path.join(args.base_path, f'test_{part1}_val_{part2}'))

    emissions, masks, probs = get_averaged_emissions_from_arrays(checkpoint_list,input_ids,input_mask, batch_size=100)
    viterbi_paths =  run_averaged_crf(checkpoint_list,emissions,masks)



    df['p_NO'] = probs[:,0]
    df['p_SPI'] = probs[:,1]
    df['p_SPII'] = probs[:,2]
    df['p_TAT'] = probs[:,3]
    df['p_TAT'] = probs[:,3]
    df['p_TATLIPO'] = probs[:,4]
    df['p_PILIN'] = probs[:,5]
    df['p_is_SP'] = probs[:,1:].sum(axis=1)

    df['Path'] = viterbi_paths #TODO need to check whether assigning a 2d array like this works
    if args.kingdom=='eukarya':
        df['pred label'] =  df[['p_NO', 'p_is_SP']].idxmax(axis=1).apply(lambda x: {'p_is_SP': 'SP', 'p_NO':'Other'}[x])
    else:
        df['pred label'] =  df[['p_NO', 'p_SPI','p_SPII','p_TAT', 'p_TATLIPO', 'p_PILIN']].idxmax(axis=1).apply(lambda x: {'p_SPI': 'Sec/SPI',
                                                                                                   'p_SPII': 'Sec/SPII', 
                                                                                                   'p_TAT':'Tat/SPI', 
                                                                                                   'p_TATLIPO':'Tat/SPII',
                                                                                                   'p_PILIN':'Sec/SPIII',
                                                                                                   'p_NO':'Other'}[x])
    
    #df = df.drop(['Sequence', 'Signal peptide'], axis=1)
    df.to_csv(args.output_file)

if __name__ == '__main__':
    main()
