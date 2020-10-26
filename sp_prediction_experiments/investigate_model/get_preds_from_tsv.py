import pandas as pd
import numpy as np
import torch
import argparse
from scipy import stats
import sys
import os
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet

from ensemble_utils import run_data_ensemble

from train_scripts.utils.signalp_dataset import LargeCRFPartitionDataset, SIGNALP_GLOBAL_LABEL_DICT, EXTENDED_VOCAB, SIGNALP_KINGDOM_DICT
from train_scripts.downstream_tasks.metrics_utils import tagged_seq_to_cs_multiclass
from models.sp_tagging_prottrans import BertSequenceTaggingCRF, ProteinBertTokenizer
from models.multi_crf_bert import BertSequenceTaggingCRF as BertSequenceTaggingMultilabelCRF
backtranslate_tokens = [x.split('_')[-1] for x in EXTENDED_VOCAB]




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get ensemble predictions')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--base_path', type=str, default='/work3/felteu/tagging_checkpoints/bert_crossval/')
    parser.add_argument('--output_path', type=str,default='/work3/felteu/preds')
    #parser.add_argument('--multi_label', action='store_true', help='Use Silas CRF BERT')
    parser.add_argument('--kingdom', type=str, default='eukarya', const='eukarya', nargs='?', choices=['eukarya', 'archaea','positive', 'negative'] )
    args = parser.parse_args()


    out_name = args.data_path.split('/')[-1].rstrip('.tsv') + '_preds.csv'
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


    tokenizer = ProteinBertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
    df = pd.read_csv(args.data_path, sep='\t')

    # tokenizer.tokenizer.pad_token =0
    seqs = df['Sequence'].apply(lambda x: x[:70]) #truncate
    tokenized = seqs.apply(lambda x: tokenizer.encode(x))
    tokenized =  tokenized.apply(lambda x: x + [0] * (72-len(x))) #pad
    input_ids = np.vstack(tokenized)

    input_mask = (input_ids>0) *1

    kingdom_id =  SIGNALP_KINGDOM_DICT[args.kingdom.upper()] #SIGNALP_KINGDOM_DICT = {'EUKARYA': 0, 'POSITIVE':1, 'NEGATIVE':2, 'ARCHAEA':3}
    print(f'Kingdom is {args.kingdom}, {kingdom_id}')

    model_input = (input_ids, input_mask, kingdom_id)


    res = run_data_ensemble(BertSequenceTaggingCRF, base_path='/work3/felteu/tagging_checkpoints/bert_crossval/', data_array=model_input, do_not_average=True)
    probs, paths, model_names = res

    #get cs
    pred_cs = []
    for path in paths:
        cs = tagged_seq_to_cs_multiclass(path, sp_tokens = [3,7,11])
        pred_cs.append(cs)

    #get label and probs
    probs =  np.stack(probs).mean(axis=0)
    pred_label = probs.argmax(axis=1)

    pred_cs = np.array(pred_cs) #shape n_models, batch_size
    #TODO need to fix mode to ignore -1 when global label is SP.
    pred_cs = pred_cs.astype(float)  
    pred_cs[pred_cs == -1] = np.nan
    mode, count = stats.mode(pred_cs,axis=0, nan_policy='omit')
    pred_cs = mode.astype(int)



    df['pred CS'] = pred_cs.squeeze()
    df['p_NO'] = probs[:,0]
    df['p_SPI'] = probs[:,1]
    df['p_SPII'] = probs[:,2]
    df['p_TAT'] = probs[:,3]
    df['p_is_SP'] = probs[:,1:].sum(axis=1)

    if args.kingdom=='eukarya':
        df['pred label'] =  df[['p_NO', 'p_is_SP']].idxmax(axis=1).apply(lambda x: {'p_is_SP': 'SP', 'p_NO':'Other'}[x])
    else:
        df['pred label'] =  df[['p_NO', 'p_SPI','p_SPII','p_TAT']].idxmax(axis=1).apply(lambda x: {'p_SPI': 'SPI',
                                                                                                   'p_SPII': 'SPII', 
                                                                                                   'p_TAT':'TAT', 
                                                                                                   'p_NO':'Other'}[x])
    
    df = df.drop(['Sequence', 'Signal peptide'], axis=1)
    df.to_csv(os.path.join(args.output_path, out_name))




