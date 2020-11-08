'''
Script that produces the training dataset for SignalP 6.0.
For new sequences, takes care of label generation.

Input files: - Graph-part result with partition assignment for each sequence
             - train_data_updated.fasta
               The SignalP5.0 dataset, with kingdoms corrected and TATLIPO+PILIN sequences added. TATLIPO also reclassified.
               Also new sequences added to it. This file is used to run graph-part.
               As 2-line fasta, labels removed
             - sp3_bacteria.csv
             - sp3_archaea.csv
             - tatlipo.fasta 3-line fasta file of all the TATLIPO sequences with manually fixed labels
             - train_set.fasta Original dataset from SignalP 5

#TODO transmembrane handling for pilins

'''
import numpy as np
import pandas as pd
import re
from typing import List, Tuple


def get_sp3_label(sequence, kingdom):
    '''make a SignalP label sequence
    All the sequences used here only have 1 TM, so did not implement multi-tm handling here (need to put I between TMs and O again after)
    expects tm_indices as 1-based indexing, not 0 (as in uniprot), so subtracts 1
    '''

    if kingdom=='ARCHAEA':
        sp_end = sequence.find('Q') -1

    elif kingdom in ['POSITIVE', 'NEGATIVE'] :
        pattern = r'(?<=[KRHEQSTAG])G[FYLIVM][ST][LT][LIVP]E[LIVMFWSTAG]'
        sp_end = re.search(pattern, sequence)
        if sp_end is not None:
            sp_end = sp_end.start()
        #there is one FN in the set - build another pattern based on this one.
        else:
        #manual match                         RG SLLEM
        #raw sequence MRVARLPLLH PHRAAPVVRR QLRGSSLLEM LLVIALIALA GVLAAAALTG
            pattern = r'(?<=[KRHEQSTAG])G[FYLIVMS][ST][LT][LIVP]E[LIVMFWSTAG]'
            sp_end = re.search(pattern, sequence)
            sp_end = sp_end.start()
    else:
        raise NotImplementedError(kingdom)

    seq_len = len(sequence)
    sp_len = sp_end+1
    labels = ['P'] * sp_len
    labels = labels + ['O'] * (seq_len-len(labels))

    labels = np.array(labels) #needs to be array so assignment to indices works

    assert len(labels) == seq_len
    return ''.join(labels)


if __name__ == '__main__':


    ## Load partition assignments
    df_part = pd.read_csv('/work3/felteu/signalp6_parts_02.csv') #AC,priority,label-val,between_connectivity,cluster
    df_part = df_part.set_index('AC')


    ## Load old dataset

    with open('../signalp_original_data/train_set.fasta', 'r') as f:
        lines = f.read().splitlines()
        identifiers = lines[::3]
        sequences = lines[1::3]
        labels = lines[2::3]

    accids = [x.strip('>').split('|')[0] for x in identifiers]
    df_old_labels =  pd.DataFrame.from_dict({'acc': accids, 'label':labels})
    df_old_labels =  df_old_labels.set_index('acc')


    ## Load tatlipo dataset

    with open('identify_tatlipo/tatlipo.fasta', 'r') as f:
        lines = f.read().splitlines()
        identifiers = lines[::3]
        sequences = lines[1::3]
        labels = lines[2::3]

    accids = [x.strip('>').split('|')[0] for x in identifiers]
    df_tatlipo =  pd.DataFrame.from_dict({'acc': accids, 'label':labels})
    df_tatlipo =  df_tatlipo.set_index('acc')


    ## Load additional sequences to be added to dataset


    ## Load new 2-line fasta file
    with open('full_updated_data_seqs_only.fasta', 'r') as f:
        lines = f.read().splitlines()
        identifiers = lines[::2]
        sequences = lines[1::2]


    def get_label(sp_type, acc=None, sequence=None, kingdom=None):
        if sp_type=='NO_SP':
            assert acc is not None, 'This case needs accession id.'
            labels = df_old_labels.loc[acc]['label']
        elif sp_type=='SP':
            assert acc is not None, 'This case needs accession id.'
            labels = df_old_labels.loc[acc]['label']
        elif sp_type=='LIPO':
            assert acc is not None, 'This case needs accession id.'
            labels = df_old_labels.loc[acc]['label']
        elif sp_type=='TAT':
            assert acc is not None, 'This case needs accession id.'
            labels = df_old_labels.loc[acc]['label']
        elif sp_type=='TATLIPO':
            assert acc is not None, 'This case needs accession id.'
            labels = df_tatlipo.loc[acc]['label']
        elif sp_type=='PILIN':
            assert sequence is not None, 'This case needs the sequence.'
            labels = get_sp3_label(sequence, kingdom)
        else:
            raise NotImplementedError(sp_type)

        return labels


    ## Iterate over all sequences, create their label, and write to three-line fasta
    with open('signalp_6_train_set.fasta', 'w') as f:

        for identifier, sequence in zip(identifiers,sequences):

            #parse header and get partition ID
            acc, kingdom, typ = identifier.lstrip('>').split('|')
            try:
                part = df_part.loc[acc]['cluster']

                #make label
                label = get_label(typ,acc,sequence,kingdom)
                #write to file
                f.write('>'+acc+'|'+kingdom+'|'+typ+'|'+str(int(part))+'\n')
                f.write(sequence + '\n')
                f.write(label + '\n')
                
            except KeyError:
                print(f'{acc} was removed by Graph-Part. Skip.')
                next
            







