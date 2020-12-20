'''
Script to compute metrics from benchmark set predictions, creates all the tables of 
the SignalP5 supplementary material.
Need prediction files from Kostas + Jose (SignalP5 raw preds)
Optionally, add new Bert pred file. Then we update the class memberships.
Otherwise reproduce tables as they were in SignalP5.

We work with 0-indexed CS here, because for the metrics it makes no difference.
When reporting CS positions, add +1 to both true and pred CS.
'''
import sys
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") 
from train_scripts.downstream_tasks.metrics_utils import compute_precision, compute_recall, mask_cs, compute_mcc
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Tuple
from pathlib import Path
import numpy as np
import ast
import os


benchmark_set_fasta_path = '../data/signal_peptides/signalp_original_data/benchmark_set.fasta'
results_base_dir = 'experiments_results/benchmark_performances_signalp5_paper/'
out_dir = 'experiments_results/benchmark_performances_signalp5_paper'#recomputed'
update_benchmark_set = False
bert_file_path = 'experiments_results/signalp_6_model/viterbi_paths.csv'
type_conversion_dict = {'NO_SP':0, 'SP':1, 'LIPO':2, 'TAT':3, 'TATLIPO':4, 'PILIN':5, np.nan: np.nan}
prediction_conversion_dict = {'None':0, 'Sec':1, 'Lipo':2, 'Tat':3, np.nan: np.nan}


# set up some utilities to read the data
    
def parse_threeline_fasta(filepath: Union[str, Path]) -> Tuple[str,str]:

    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
        identifiers = lines[::3]
        sequences = lines[1::3]
        labels = lines[2::3]

    return identifiers, sequences

def parse_twoline_fasta(filepath: Union[str, Path]) -> Tuple[str,str]:

    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
        identifiers = lines[::2]
        sequences = lines[1::2]

    return identifiers, sequences

def get_sp_len(token_ids):
    '''Find the CS in multi-tag bert'''
    token_ids = np.array(token_ids)
    sp_pos = np.isin(token_ids, [5,11,19,26,31])
    pos_indices = np.where(sp_pos)[0]

    return pos_indices.max() if len(pos_indices) >0 else -1

def get_cs_and_type(label_sequence):
    '''Find CS from label sequence and infer type if has CS
    Use the class labels used in the SignalP5 preds
    CS is last idx that is SP (0-based indexing)'''
    if 'C' in label_sequence:
        cs = label_sequence.rfind('C')
        spclass = 'Default' #for signalp5, gets global preds from other file anyway
    elif 'S' in label_sequence:
        cs = label_sequence.rfind('S')
        spclass =  'Sec'
    elif 'T' in label_sequence:
        cs = label_sequence.rfind('T')
        spclass = 'Tat'
    elif 'L' in label_sequence:
        cs = label_sequence.rfind('L')
        spclass = 'Lipo'
    else:
        cs = -1
        spclass = 'None'
        
    return cs, spclass



# function to parse all the files into pandas dfs
def parse_all_files(benchmark_set_fasta_path: str, results_base_dir: str):
    '''parse results files to two dfs with class and cs predictions.
    benchmark_set_fasta_path: path to benchmark_set.fasta
    results_base_dir: path to dir that contains all the files from Kostas and Jose.
    '''
    with open(benchmark_set_fasta_path, 'r') as f:
        lines = f.read().splitlines()
        identifiers = lines[::3]
        sequences = lines[1::3]
        labels = lines[2::3]

        # parse headers
        entries = []
        kingdoms = []
        types= []
        for head in identifiers:
            entry, kingdom, sptype, _ = head.split('|')
            entries.append(entry)
            kingdoms.append(kingdom)
            types.append(sptype)
            

        
        #parse true CS
        cs = [get_cs_and_type(x)[0] for x in labels]
        
        #make df
        df_class =  pd.DataFrame.from_dict({'Entry':entries,'Kingdom':kingdoms,'Type':types}).set_index('Entry')
        df_cs =  pd.DataFrame.from_dict({'Entry':entries,'Kingdom':kingdoms,'Type':types, 'True cleavage site': cs}).set_index('Entry')
    

    # parse the other tools
    files_to_parse = ['DEEPSIG_converted.res', 'LipoP_converted.res', 'PHILIUS.res', 
                    'PHOBIUS.res', 'PolyPhobius.res', 'PRED-LIPO.res', 'PRED-SIGNAL.res', 'PRED-TAT.res',
                    'SIGNAL_CF_converted.res', 'Signal3Lv2_converted.res', 'SOSUIsignal_converted.res',
                    'SPEPlip_converted.res', 'SPOCTOPUS.res', 'TatP_converted.res', 'TOPCONS2.res']
    for file in files_to_parse:
        headers, seqs = parse_twoline_fasta(os.path.join(results_base_dir,file))
        #make two pandas series to enable joining
        df = pd.DataFrame.from_dict({'Entry':headers,'labels':seqs})
        df = df.drop_duplicates(subset='Entry').set_index('Entry')    
        parsed = df['labels'].apply(get_cs_and_type)
        cs = parsed.apply(lambda x: x[0])
        cls = parsed.apply(lambda x: x[1])
        
        name = file.split('.')[0].split('_converted')[0]#.rstrip('_converted')
        cls.name = name
        cs.name = name
            
        df_class = df_class.join(cls)
        df_cs  = df_cs.join(cs)
    
    
    ## parse TATFIND
    with open(os.path.join(results_base_dir,'TATFIND.res'), 'r') as f:
            lines = f.read().splitlines()
            lines = [x for x in lines if x.startswith('Results')]
            #'Results for A0A068FIK2: FALSE'
            entries = ['>' + x.split(' ')[2].rstrip(':') for x in lines]
            preds = ['Tat' if x.split(' ')[-1] == 'TRUE' else 'None' for x in lines]
            
            df = pd.DataFrame.from_dict({'Entry': entries, 'TATFIND': preds})
            df = df.drop_duplicates(subset='Entry').set_index('Entry')   
            
    df_class = df_class.join(df['TATFIND'])


    # parse SignalP5 preds
    headers, seqs = parse_threeline_fasta(os.path.join(results_base_dir,'prediction_amino_acid_new.txt'))
    df = pd.DataFrame.from_dict({'Entry':headers,'labels':seqs}).set_index('Entry')
    parsed = df['labels'].apply(get_cs_and_type)
    cs = parsed.apply(lambda x: x[0])
    cs.name = 'SignalP5'
    df_cs  = df_cs.join(cs)

    df = pd.read_csv(os.path.join(results_base_dir,'prediction_sequence_new.txt'), sep='\t')
    df['Entry'] = ['>'+x for x in df['ID']]

    df = df.set_index('Entry').rename({'Prediction':'SignalP5'}, axis=1)
    df_class = df_class.join(df['SignalP5'])

    return df_class, df_cs


def parse_bert_file(bert_file_path:str):
    '''Parse a file produced by averaged_viterbi_decoding.py
        df_class : Kingdom	Type	Bert
        df_cs    : Kingdom	Type	Bert	True cleavage site
    '''
    df = pd.read_csv(bert_file_path, index_col=0, usecols=['Entry', 'Kingdom', 'Type', 'Path', 'Pred label', 'True path'])
    df['Path'] = df['Path'].apply(ast.literal_eval)
    df['Bert'] = df['Path'].apply(lambda x: get_sp_len(x))

    df_class =  df.drop(['True path', 'Path', 'Bert'], axis=1)
    df_class = df_class.rename({'Pred label':'Bert'}, axis=1)

    df_cs =  df.drop(['Path', 'True path', 'Pred label'], axis=1)
    df_cs['True cleavage site'] = df['True path'].apply(lambda x: get_cs_and_type(x)[0])

    return df_class, df_cs


# make MCC tables
def make_mcc_tables(df_class: pd.DataFrame, out_dir: str) -> None:
    
    for sp_class in ['SP','LIPO','TAT']:
        mccs = {}
        positive_class = type_conversion_dict[sp_class]
        for org in ['ARCHAEA', 'EUKARYA','NEGATIVE','POSITIVE']:
            mccs[org+ ' mcc2'] ={}
            mccs[org+ ' mcc1'] ={}
            org_df = df_class.loc[df_class['Kingdom']==org]

            for tool in org_df.columns[2:]:
                targets = org_df['Type_int'].values
                preds = org_df[tool].values
                mccs[org+ ' mcc2'][tool] = compute_mcc(targets,preds , label_positive = positive_class)
                mcc1_idx = np.isin(targets, [positive_class, 0])
                mccs[org+ ' mcc1'][tool] = compute_mcc(targets[mcc1_idx] ,preds[mcc1_idx] , label_positive = positive_class)


        df = pd.DataFrame.from_dict(mccs)
        df.to_csv(os.path.join(out_dir, f'{sp_class}_benchmark_mcc.csv'))


# make precision, recall tables

# precision, recall
def make_precision_recall_tables(df_class: pd.DataFrame, df_cs: pd.DataFrame, out_dir:str) -> None:

    for sp_class in ['SP','LIPO','TAT']:

        positive_class = type_conversion_dict[sp_class]
        precisions = {}
        recalls = {}
        for org in ['ARCHAEA', 'EUKARYA','NEGATIVE','POSITIVE']:

            org_cls_df = df_class.loc[df_class['Kingdom']==org]
            org_cs_df = df_cs.loc[df_cs['Kingdom']==org]

            ## all need to match:  global_labels, pred_labels, label_to_keep, negative_label):

            for window in [0,1,2,3]:
                precisions[org + ' ' + str(window)] = {}
                recalls[org + ' ' + str(window)] = {}
                for tool in org_cs_df.columns:

                    if tool in ['True cleavage site', 'Kingdom', 'Type']:
                        continue # skip non-predictor columns

                    # get all the targets and predictions - do not subset for class
                    targets = org_cls_df['Type_int'].values
                    preds = org_cls_df[tool].values
                    cs_preds = org_cs_df[tool].values
                    cs_targets =  org_cs_df['True cleavage site'].values

                    # now mask other types (set their cs target to -1)
                    # and remove samples that were correctly predicted as another class
                    true_CS, pred_CS = mask_cs(cs_targets, cs_preds, targets, preds,label_positive=positive_class, label_negative=0)

                    precisions[org + ' ' + str(window)][tool] = compute_precision(true_CS, pred_CS, window_size = window)
                    recalls[org + ' ' + str(window)][tool] = compute_recall(true_CS, pred_CS, window_size = window)


        #print(pd.DataFrame.from_dict(precisions).drop('True cleavage site', axis=0).idxmax(axis=0))
        df = pd.DataFrame.from_dict(precisions)#.loc[['SignalP5', 'DEEPSIG']]
        df.to_csv(os.path.join(out_dir, f'{sp_class}_benchmark_recalls.csv'))
        df = pd.DataFrame.from_dict(recalls)
        df.to_csv(os.path.join(out_dir, f'{sp_class}_benchmark_precisions.csv'))


def main():

    os.makedirs(out_dir,exist_ok=True)

    ## parse data
    df_class, df_cs =  parse_all_files(benchmark_set_fasta_path,results_base_dir)



    ## process for metric functions: need integer label ids
    for col in df_class.columns[1:]:
        if col == 'Type':
            df_class['Type_int'] =  df_class[col].apply(lambda x: type_conversion_dict[x])
        else:
            df_class[col] =  df_class[col].apply(lambda x: prediction_conversion_dict[x])

    #        df_class : Kingdom	Type	Bert
    #        df_cs    : Kingdom	Type	Bert	True cleavage site
    # optional: add signalp6 results and reclassify/remove here
    if update_benchmark_set == True:
        df_class_bert, df_cs_bert = parse_bert_file(bert_file_path)
        # inner join of the Bert preds and the old benchmark preds.
        # use Kingdom and Type column of the Bert df -> updated gram-positive and tatlipo classification
        df_class = df_class.drop(['Kingdom', 'Type'],axis=1).join(df_class_bert, how='inner')
        df_class['Type_int'] =  df_class['Type'].apply(lambda x: type_conversion_dict[x])

        df_cs = df_cs.drop(['Kingdom', 'Type', 'True cleavage site'],axis=1).join(df_cs_bert, how='inner')


    ## make tables
    make_mcc_tables(df_class, out_dir=out_dir)
    make_precision_recall_tables(df_class, df_cs, out_dir=out_dir)


if __name__ == '__main__':
    main()