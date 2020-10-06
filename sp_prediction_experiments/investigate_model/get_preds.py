from ensemble_utils import run_data_ensemble
import pandas as pd
import torch
import argparse
import sys
import os
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet

from train_scripts.utils.signalp_dataset import LargeCRFPartitionDataset, SIGNALP_GLOBAL_LABEL_DICT, SIGNALP_VOCAB
from models.sp_tagging_prottrans import BertSequenceTaggingCRF, ProteinBertTokenizer



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get ensemble predictions')
    parser.add_argument('--data_path', type=str, default='/work3/felteu/data/signalp_5_data/full/train_set.fasta')
    parser.add_argument('--base_path', type=str, default='/work3/felteu/tagging_checkpoints/bert_crossval/')
    parser.add_argument('--output_path', type=str,default='/work3/felteu/preds')
    args = parser.parse_args()
    #get tokenizer
    tokenizer = ProteinBertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)

    #load dataset
    ds = LargeCRFPartitionDataset(args.data_path,tokenizer=tokenizer,add_special_tokens=True,return_kingdom_ids=True)
    #predict
    dl = torch.utils.data.DataLoader(ds, collate_fn=ds.collate_fn, batch_size=100)

    all_losses, all_global_targets, all_global_probs, all_targets, all_pos_preds = run_data_ensemble(BertSequenceTaggingCRF,args.base_path, dl)


    #process loss - average over sequence length
    #all_losses = all_losses.mean(axis=1)

    #make a dataframe
    ds.identifiers #n_sequences
    all_losses #n_sequences
    all_global_targets #n_sequences
    all_global_probs #n_sequences x 4
    all_pos_preds #n_sequences x 7

    cs = tagged_seq_to_cs_multiclass(all_pos_preds, sp_tokens = [3,7,11])

    from IPython import embed
    embed()

    df = pd.DataFrame.from_dict({'loss': all_losses, 
                                 'identifiers':ds.identifiers,
                                 'target':all_global_targets,
                                 'CS': cs,
                                 'p_NO' : all_global_probs[:,0],
                                 'p_SPI': all_global_probs[:,1],
                                 'p_SPII': all_global_probs[:,2],
                                 'p_TAT': all_global_probs[:,3],
                                 
                                 })


df[['ID', 'Kingdom', 'Type', 'Partition']] = df['identifiers'].str.lstrip('>').str.split('|', expand=True)
df = df.set_index('ID')

#rearrange
df = df[['Kingdom', 'Type', 'Partition', 'loss', 'p_NO', 'p_SPI', 'p_SPII', 'p_TAT', 'CS', 'target']]
df.to_csv('complete_set_probs_loss.csv')


##code to run plasmodium - 
def run_plasmodium():
    import pandas as pd
    import numpy as np
    from models.sp_tagging_prottrans import BertSequenceTaggingCRF, ProteinBertTokenizer
    from sp_prediction_experiments.investigate_model.ensemble_utils import run_data_ensemble

    tokenizer = ProteinBertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)

    df = pd.read_csv('sp_prediction_experiments/uniprot_manual_plasmodium_sps.tsv', sep='\t')

    tokenized = df['Sequence'].apply(lambda x: tokenizer.encode(x[:70]))
    model_input = np.vstack(tokenized)

    res = run_data_ensemble(BertSequenceTaggingCRF, dataloader='x', base_path='/work3/felteu/tagging_checkpoints/bert_crossval/', data_array=model_input)
