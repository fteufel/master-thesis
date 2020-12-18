'''
Predict sequences from a .tsv file with all 20 models and average.
This does not do viterbi decoding averaging.
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


def run_data_array(model: torch.nn.Module, input_ids:  np.ndarray, input_mask: np.ndarray, batch_size = 100):
    '''Run numpy array dataset. This function takes care of batching
    and putting the outputs back together.
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_global_probs = []
    all_pos_preds = []

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
        all_pos_preds.append(pos_preds.detach().cpu().numpy())

        b_start = b_start + batch_size
        b_end = b_end + batch_size


    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)

    return all_global_probs, all_pos_preds



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

    if do_not_average:
        return output_obj + [name_list]

    #average the predictions
    avg_list = []
    for obj in output_obj:
        #identify type - does not need to be tensor
        if type(obj[0]) == torch.Tensor:
            avg = torch.stack(obj).float().mean(axis=0) #call float to avoid error when dealing with longtensors
        elif type(obj[0]) == np.ndarray:
            avg = np.stack(obj).mean(axis=0)
        else:
            raise NotImplementedError

        avg_list.append(avg)

    return avg_list


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


    model = BertSequenceTaggingCRF
    res = run_data_ensemble(model, base_path=args.base_path, input_ids=input_ids, input_mask=input_mask, do_not_average=True, partitions=list(range(args.n_partitions)))
    probs, paths, model_names = res


    #get label and probs
    probs =  np.stack(probs).mean(axis=0)
    pred_label = probs.argmax(axis=1)


    df['p_NO'] = probs[:,0]
    df['p_SPI'] = probs[:,1]
    df['p_SPII'] = probs[:,2]
    df['p_TAT'] = probs[:,3]
    df['p_TAT'] = probs[:,3]
    df['p_TATLIPO'] = probs[:,4]
    df['p_PILIN'] = probs[:,5]
    df['p_is_SP'] = probs[:,1:].sum(axis=1)

    if args.kingdom=='eukarya':
        df['pred label'] =  df[['p_NO', 'p_is_SP']].idxmax(axis=1).apply(lambda x: {'p_is_SP': 'SP', 'p_NO':'Other'}[x])
    else:
        df['pred label'] =  df[['p_NO', 'p_SPI','p_SPII','p_TAT', 'p_TATLIPO', 'p_PILIN']].idxmax(axis=1).apply(lambda x: {'p_SPI': 'Sec/SPI',
                                                                                                   'p_SPII': 'Sec/SPII', 
                                                                                                   'p_TAT':'Tat/SPI', 
                                                                                                   'p_TATLIPO':'Tat/SPII',
                                                                                                   'p_PILIN':'Sec/SPIII',
                                                                                                   'p_NO':'Other'}[x])
    
    df = df.drop(['Sequence', 'Signal peptide'], axis=1)
    df.to_csv(args.output_file)

if __name__ == '__main__':
    main()
