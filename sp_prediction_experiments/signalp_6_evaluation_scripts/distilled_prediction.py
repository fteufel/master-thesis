'''
Predict sequences from a .tsv file with a single model checkpoint.
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


def get_preds(checkpoint: str,  input_ids:  np.ndarray, 
                                 input_mask: np.ndarray, batch_size = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Run sequences through all models in list.
    Return only the emissions (Bert+linear projection) and average those.
    Also returns input masks, as CRF needs them when decoding.
    '''

    # Load model and get emissions
    model = BertSequenceTaggingCRF.from_pretrained(checkpoint)
    model.to(device)
    model.eval()


    probs_batched = []
    pos_probs_batched =[]
    pos_preds_batched = []

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
            global_probs, pos_probs, pos_preds = model(data, input_mask=mask)

            probs_batched.append(global_probs.cpu())
            pos_probs_batched.append(pos_probs.cpu())
            pos_preds_batched.append(pos_preds.cpu())
        
        b_start = b_start + batch_size
        b_end = b_end + batch_size

    #covert to CPU tensors after forward pass. Save memory.
    model_probs = torch.cat(probs_batched).detach().cpu().numpy()
    model_pos_probs = torch.cat(pos_probs_batched).detach().cpu().numpy()
    model_pos_preds = torch.cat(pos_preds_batched).detach().cpu().numpy()


    return model_probs, model_pos_probs, model_pos_preds



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
    parser.add_argument('--checkpoint', type=str, default='/work3/felteu/tagging_checkpoints/signalp_6_distilled')
    parser.add_argument('--output_file', type=str,default='ensemble_predictions.csv')
    parser.add_argument('--kingdom', type=str, default='EUKARYA', const='EUKARYA', nargs='?', choices=['EUKARYA', 'ARCHAEA','POSITIVE', 'NEGATIVE'] )

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

        seqs = df['Sequence'].apply(lambda x: x[:70]) #truncate
        tokenized = seqs.apply(lambda x: tokenizer.encode(x, kingdom_id=args.kingdom))
        tokenized =  tokenized.apply(lambda x: x + [0] * (73-len(x))) #pad #73 = 70 +cls + kingdom + sep
    
    input_ids = np.vstack(tokenized)
    input_mask = (input_ids>0) *1



    probs, pos_probs, pos_preds = get_preds(args.checkpoint,input_ids,input_mask, batch_size=100)



    df['p_NO'] = probs[:,0]
    df['p_SPI'] = probs[:,1]
    df['p_SPII'] = probs[:,2]
    df['p_TAT'] = probs[:,3]
    df['p_TAT'] = probs[:,3]
    df['p_TATLIPO'] = probs[:,4]
    df['p_PILIN'] = probs[:,5]
    df['p_is_SP'] = probs[:,1:].sum(axis=1)

    df['Path'] = pos_preds.tolist()
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
