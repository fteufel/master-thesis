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
import h5py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def memory_efficient_mean(tensor_list: List[torch.Tensor]):
    '''
    torch.stack().mean() goes OOM for large datasets,
    although the list of tensors fits memory. 
    Use this as a workaround.
    '''
    with torch.no_grad():
        sum_tensor = torch.zeros_like(tensor_list[0])
        for t in tensor_list:
            sum_tensor = sum_tensor + t

        return sum_tensor / len(tensor_list)

def get_averaged_emissions_from_arrays(model_checkpoint_list: List[str],  input_ids:  np.ndarray, 
                                 input_mask: np.ndarray, batch_size = 500) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Run sequences through all models in list.
    Return only the emissions (Bert+linear projection) and average those.
    Also returns input masks, as CRF needs them when decoding.
    '''

    emission_list = []
    global_probs_list = []
    pos_probs_list = []
    for checkpoint in tqdm(model_checkpoint_list):

        # Load model and get emissions
        model = BertSequenceTaggingCRF.from_pretrained(checkpoint)
        model.to(device)
        model.eval()

        emissions_batched = []
        masks_batched = []
        probs_batched = []
        pos_probs_batched =[]

        b_start = 0
        b_end = batch_size


        for b_start in tqdm(range(0,len(input_ids),batch_size),leave=False):

            b_end=b_start+batch_size
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
                pos_probs_batched.append(pos_probs.cpu())
            
        #covert to CPU tensors after forward pass. Save memory.
        model_emissions = torch.cat(emissions_batched).detach().cpu()
        emission_list.append(model_emissions)
        model_pos_probs = torch.cat(pos_probs_batched).detach().cpu()
        pos_probs_list.append(model_pos_probs)
        masks =  torch.cat(masks_batched) # gathering mask in inner loop is sufficent, same every time.
        model_probs = torch.cat(probs_batched).detach().cpu()
        global_probs_list.append(model_probs)

    # Average the emissions
    emission_list_mean = memory_efficient_mean(emission_list)
    global_probs_list_mean = memory_efficient_mean(global_probs_list)
    pos_probs_list_mean = memory_efficient_mean(pos_probs_list)
    return emission_list_mean, masks, global_probs_list_mean, pos_probs_list_mean



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
        pos_probs = model.crf.compute_marginal_probabilities(emissions=emissions, mask=input_mask.byte())

    return viterbi_paths, pos_probs

def threeline_fasta_to_df(file_path):
    '''For convenience, convert three-line fasta files to df format,
    so we can reuse all the functions. '''

    ds = RegionCRFDataset(file_path)
    seqs = ds.sequences
    labs = ds.labels
    types = ds.global_labels
    ids = ds.identifiers
    kingdom_id = ds.kingdom_ids

    df =  pd.DataFrame.from_dict({'Sequence':seqs,'label':labs, 'type':types,'id':ids, 'kingdom':kingdom_id})
    return df

def main():

    parser = argparse.ArgumentParser('Get ensemble predictions')
    parser.add_argument('--data', type=str)
    parser.add_argument('--base_path', type=str, default='/work3/felteu/tagging_checkpoints/signalp_6/')
    parser.add_argument('--output_file', type=str,default='ensemble_predictions.csv')
    parser.add_argument('--kingdom', type=str, default='EUKARYA', const='EUKARYA', nargs='?', choices=['EUKARYA', 'ARCHAEA','POSITIVE', 'NEGATIVE'] )
    parser.add_argument('--n_partitions', type=int, default=3, help='Number of partitions, for loading the checkpoints and datasets.')
    parser.add_argument('--make_distillation_targets', action='store_true', help='save probabilities as pickle.')

    args = parser.parse_args()

    args = parser.parse_args()

    tokenizer=ProteinBertTokenizer.from_pretrained("/zhome/1d/8/153438/experiments/master-thesis/resources/vocab_with_kingdom", do_lower_case=False)

    if args.data.endswith('.fasta'):
        print('Assuming three-line fasta file. 2-line fasta handling not yet implemented')
        df =  threeline_fasta_to_df(args.data)
        seqs = df['Sequence'].apply(lambda x: x[:70]) #truncate
        tokenized = [tokenizer.encode(x, kd) for x,kd in zip(seqs,df.kingdom)]
        tokenized = [x +[0] * (73-len(x)) for x in tokenized] #pad

    else:
        df = pd.read_csv(args.data, sep='\t')

        seqs = df['Reference sequence'].apply(lambda x: x[:70]) #truncate
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

    emissions, masks, probs, pos_probs = get_averaged_emissions_from_arrays(checkpoint_list,input_ids,input_mask)
    from IPython import embed
    embed()

    #save the probs as hdf5
    if args.make_distillation_targets:
        print('Saving predicted probabilities as hdf5')
        with h5py.File(args.output_file, "w") as f:
            f.create_dataset('input_ids', data=input_ids)
            f.create_dataset('input_mask', data=input_mask)
            f.create_dataset('pos_probs', data=pos_probs)
            true_labels = df['type'].apply(lambda x:{'NO_SP': 0, 'SP': 1, 'LIPO': 2, 'TAT': 3, 'TATLIPO': 4, 'PILIN':5}[x]).values
            f.create_dataset('type', data=true_labels)#data=probs.argmax(axis=1))
        

    #add results to df and save as csv
    else:
        viterbi_paths, pos_probs_crf =  run_averaged_crf(checkpoint_list,emissions,masks)

        df['p_NO'] = probs[:,0]
        df['p_SPI'] = probs[:,1]
        df['p_SPII'] = probs[:,2]
        df['p_TAT'] = probs[:,3]
        df['p_TAT'] = probs[:,3]
        df['p_TATLIPO'] = probs[:,4]
        df['p_PILIN'] = probs[:,5]
        df['p_is_SP'] = probs[:,1:].sum(axis=1)

        df['Path'] = viterbi_paths
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
