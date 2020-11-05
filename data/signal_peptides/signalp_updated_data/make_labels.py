'''
Script to make labels for new dataset

TAT, LIPO, SP, NO_SP: reuse old labels

LIPIN, TATLIPO: make labels from .tsv uniprot files
'''
import numpy as np
import pandas as pd
import re
from typing import List, Tuple


def generate_label_sequence(sp_len: int, tm_indices: List[Tuple[int,int]] = None, seq_len=70, sp_symbol = 'S'):
    '''make a SignalP label sequence
    All the sequences used here only have 1 TM, so did not implement multi-tm handling here (need to put I between TMs and O again after)
    expects tm_indices as 1-based indexing, not 0 (as in uniprot), so subtracts 1
    '''
    if len(tm_indices)>1:
        raise NotImplementedError

    labels = [sp_symbol] * sp_len
    labels = labels + ['O'] * (seq_len-len(labels))

    labels = np.array(labels) #needs to be array so assignment to indices works

    for tm in tm_indices:
        tm_start = int(tm[0]) -1
        tm_end = int(tm[1]) #don't subtract here for correct slicing behaviour
        labels[tm_start:tm_end] = 'M'
        labels[tm_end:] = 'I'

    assert len(labels) == seq_len
    return ''.join(labels)



def get_transmem_indices(df):
    '''Parse Uniprot field Transmembrane'''
    transmems =   df['Transmembrane'].str.split(';')

    #Filter for list items that start with TRANSMEM, those have the numbers
    transmems = transmems.apply(lambda x: [y for y in x if y.startswith('TRANSMEM')] if x is not np.nan else None)
    transmems = transmems.apply(lambda x: [tuple(y.strip('TRANSMEM ').split('..')) for y in x] if x is not None else x)

    return transmems



#[KRHEQSTAG]-G-[FYLIVM]-[ST]-[LT]-[LIVP]-E-[LIVMFWSTAG] Bacteria, cleavage after G
#“QXSXEXXXL” with the Q being the +1 residue. Archaea
def find_sp3_end(sequence, kingdom):
    '''Return sp end (0-based idx of last residue that is part of SP)'''
    if kingdom=='archaea':
        sp_end = sequence.find('Q') -1

    if kingdom=='bacteria':
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

    
    return sp_end +1




if __name__ == '__main__':

    #generate SPIII labels for archaea and bacteria from uniprot data
    df = pd.read_csv('sp3_bacteria.tsv', sep='\t')
    sp_ends = df['Sequence'].apply(lambda x: find_sp3_end(x, 'bacteria'))
    tm = get_transmem_indices(df)
    df['sp_ends'] = sp_ends
    df['tm'] = tm
    df_bacteria = df.set_index('Entry')

    df = pd.read_csv('sp3_archaea.tsv', sep='\t')
    sp_ends = df['Sequence'].apply(lambda x: find_sp3_end(x, 'archaea'))
    tm = get_transmem_indices(df)
    df['sp_ends'] = sp_ends
    df['tm'] = tm
    df_archaea = df.set_index('Entry')

    df_sp3 = pd.concat([df_bacteria, df_archaea])


    #read the old fasta file with labels
    with open('../signalp_original_data/train_data.fasta', 'r') as f:
        lines = f.read().splitlines()
        identifiers = lines[::3]
        sequences = lines[1::3]
        labels = lines[2::3]

    accids = [x.strip('>').split('|')[0] for x in identifiers]
    df_old_labels =  pd.DataFrame.from_dict({'acc': accids, 'label':labels})
    df_old_labels =  df_old_labels.set_index('acc')

    #read the new fasta file with sequences and headers
    #split ids were removed, positive/negative fixed and new sequences added
    with open('full_updated_data_seqs_only.fasta', 'r') as f:
        lines = f.read().splitlines()
        identifiers = lines[::2]
        sequences = lines[1::2]

    #read the new homology partition assignments
    part_df = None

    
    #write new three-line fasta file
    with open('signalp_6_train_set.fasta', 'w') as f:

        for (identifier, sequence) in zip(identifiers, sequences):

            acc, kingdom, typ = identifier.lstrip('>').split('|')
            part = part_df[acc] #TODO add part loading

            f.write('>'+acc+'|'+kingdom+'|'+typ+'|'+part+'\n')
            f.write(sequence + '\n')

            if typ in ['NO_SP', 'SP', 'TAT', 'LIPO']:
                #can reuse old labels
                label = df_old_labels[acc]['label']
                f.write(label + '\n')

            elif typ == 'PILIN':
                #generate new labels
                s = df_sp3.loc[acc]
                label = generate_label_sequence(s[sp_ends]+1, s['tm'], seq_len = len(sequence), sp_symbol = 'P')
                f.write(label + '\n')
            
            elif typ =='TATLIPO':
                #need to update old labels or generate new labels
                try:
                    label = df_old_labels[acc]['label']
                    lipobox_pattern = r'[GAS]C'
                    f.write(label + '\n')
                except KeyError:
                    f.write('****Fix manually' + '\n')



