'''
Script that produces the training dataset for SignalP 6.0.
For new sequences, takes care of label generation.

Input files: - Graph-part result with partition assignment for each sequence
             - full_updated_data_seqs_only.fasta
               The SignalP5.0 dataset, with kingdoms corrected and TATLIPO+PILIN sequences added. TATLIPO also reclassified.
               As 2-line fasta, labels removed
             - sp3_bacteria.csv
             - sp3_archaea.csv
             - tatlipo.fasta 3-line fasta file of all the TATLIPO sequences with manually fixed labels
             - train_set.fasta Original dataset from SignalP 5

#TODO transmembrane handling for pilins

Flow:
Iterate over sequences in PARTITIONING_DATA_FILE (what was used to generate the partition assignments)
Get header and sequence from there, add assignment from PARTITION_ASSIGNMENTS to the header
If SP/NO_SP/LIPO/TAT, get label either from ORIGINAL_DATA/EXTENSION_DATA
If TATLIPO, get label from TATLIPO_DATA
If SP3, generate labels from tm info in .csv and regex in this script.

'''
import numpy as np
import pandas as pd
import re
from typing import List, Tuple


PARTITION_ASSIGNMENTS = '/work3/felteu/class_separated_partitions/nokingdom_concatenated.csv'#'signalp_6_graphpart_partition_assignments.csv'
ORIGINAL_DATA = '../signalp_original_data/train_set.fasta'
EXTENSION_DATA = 'update_tat_sp_lipo/all_extensions.fasta'
TATLIPO_DATA = 'identify_tatlipo/tatlipo.fasta'
PARTITIONING_FASTA_FILE = 'signalp_6_seqs_only.fasta'
PILIN_DATA_BAC = 'sp3_bacteria.tsv'
PILIN_DATA_ARC = 'sp3_archaea.tsv'
OUT_FILE_NAME = 'signalp_6_train_set.fasta'


def parse_tm(tm_series: pd.Series, filter_exp = False):
    '''Parse a uniprot .tsv transmembrane column to lists of (start,end) tuples.'''
    tm_series = tm_series.copy()

    for idx, item in tm_series.iteritems():
        if type(item) == str:
            tm_annots = [x for x in item.split(';') if x.startswith('TRANSMEM')]
            tm_evidence = [x for x in item.split(';') if x.startswith('  /evidence')]

            #possibly more than one tm region annotated
            tm_region_list = []
            for annot, evidence in zip(tm_annots, tm_evidence):
                #if filter, only return experimental evidence TMs
                if filter_exp:
                    if 'ECO:0000269' in evidence:
                        start,end = annot.lstrip('TRANSMEM ').split('..')
                        tm_region_list.append( (int(start), int(end)))
                else:
                    start,end = annot.lstrip('TRANSMEM ').split('..')
                    tm_region_list.append( (int(start), int(end)))
    
            tm_series[idx] = tm_region_list

    return tm_series

def get_sp3_label(sequence, kingdom, tm_start, tm_end):
    ''' Generate label sequence for sp3 sequences.
    All seqs have a tm region annotated that we don't trust.
    Annotated it as TM anyway, will be converted in label_preprocessing to h region.
    TM_end is 1-based indexed (Uniprot)'''

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
        #Consulted PILINFIND, agrees with this pattern
            pattern = r'(?<=[KRHEQSTAG])G[FYLIVMS][ST][LT][LIVP]E[LIVMFWSTAG]'
            sp_end = re.search(pattern, sequence)
            sp_end = sp_end.start()
    else:
        raise NotImplementedError(kingdom)

    sp_len = sp_end+1
    #manual fix when regions overlap, as proposed by Kostas.
    #just add some o tags between, tms are predicted after all, not accurate
    if sp_len >= tm_start:
        tm_start = sp_len+2
    seq_len = len(sequence)
    
    labels = ['P'] * sp_len
    labels = labels+ ['O'] * (tm_start-sp_len)
    labels = labels+ ['M'] * (tm_end-tm_start)
    labels = labels + ['O'] * (seq_len-tm_end)

    labels = np.array(labels) #needs to be array so assignment to indices works


    assert len(labels) == seq_len
    return ''.join(labels)


if __name__ == '__main__':


    ## Load partition assignments
    df_part = pd.read_csv(PARTITION_ASSIGNMENTS) #AC,priority,label-val,between_connectivity,cluster
    df_part = df_part.set_index('AC')


    ## Load old dataset

    with open(ORIGINAL_DATA, 'r') as f:
        lines = f.read().splitlines()
        identifiers = lines[::3]
        sequences = lines[1::3]
        labels = lines[2::3]

    accids = [x.strip('>').split('|')[0] for x in identifiers]
    df_old_labels =  pd.DataFrame.from_dict({'acc': accids, 'label':labels})
    df_old_labels =  df_old_labels.set_index('acc')


    ## Load tatlipo dataset

    with open(TATLIPO_DATA, 'r') as f:
        lines = f.read().splitlines()
        identifiers = lines[::3]
        sequences = lines[1::3]
        labels = lines[2::3]

    accids = [x.strip('>').split('|')[0] for x in identifiers]
    df_tatlipo =  pd.DataFrame.from_dict({'acc': accids, 'label':labels})
    df_tatlipo =  df_tatlipo.set_index('acc')


    ## Load SP3 Transmembrane info
    df_sp3 = pd.read_csv(PILIN_DATA_ARC, sep='\t')
    df2 = pd.read_csv(PILIN_DATA_BAC, sep='\t')

    df_sp3 = pd.concat([df_sp3,df2]).reset_index()
    df_sp3['tm_positions'] = parse_tm(df_sp3['Transmembrane'])
    df_sp3 = df_sp3.set_index('Entry')



    ## Load additional sequences to be added to dataset
    with open(EXTENSION_DATA, 'r') as f:
        lines = f.read().splitlines()
        identifiers = lines[::3]
        sequences = lines[1::3]
        labels = lines[2::3]

    accids = [x.strip('>').split('|')[0] for x in identifiers]
    df_additional_old_classes = pd.DataFrame.from_dict({'acc': accids, 'label':labels})
    df_additional_old_classes =  df_additional_old_classes.set_index('acc')

    #This has the same format as the SignalP5.0 dataset. Just concatenate the two, then label
    #lookup works the same for old+new SP/LIPO/TAT/NO_SP
    df_old_labels = pd.concat([df_old_labels, df_additional_old_classes])

    ## Load new 2-line fasta file
    with open(PARTITIONING_FASTA_FILE, 'r') as f:
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
            assert acc is not None and sequence is not None, 'This case needs the sequence.'
            tm_info = df_sp3.loc[acc]['tm_positions']
            labels = get_sp3_label(sequence, kingdom, tm_start=tm_info[0][0], tm_end=tm_info[0][1])
            
        else:
            raise NotImplementedError(sp_type)

        return labels


    ## Iterate over all sequences, create their label, and write to three-line fasta
    with open(OUT_FILE_NAME, 'w') as f:

        for identifier, sequence in zip(identifiers,sequences):

            #parse header and get partition ID
            acc, kingdom, typ = identifier.lstrip('>').split('|')
            try:
                part = df_part.loc[acc]['cluster']

                #make label
                label = get_label(typ,acc,sequence,kingdom)

                #TODO length check
                
                #write to file
                f.write('>'+acc+'|'+kingdom+'|'+typ+'|'+str(int(float(part)))+'\n') #handle printed floats of graph-part outputs, should really be ints
                f.write(sequence + '\n')
                f.write(label + '\n')
                
            except KeyError:
                print(f'{acc} was removed by Graph-Part. Skip.')
                continue
            







