import pandas as pd
import numpy as np

def make_sequence_tags(sequences: pd.Series, tag_borders : pd.Series ) -> pd.Series:
    '''Makes sequence label string. S for Signal Peptide, P for rest of protein
        sequences: pd.Series of strings
        tag_borders: pd.Series of lists [start,end]
    '''
    assert len(sequences) == len(tag_borders)
    #make tag series, contains numpy arrays
    tags =  sequences.copy()
    tags = tags.apply(lambda x: np.array(list(x)))
    for i, seq in tags.iteritems():
        start, end = tag_borders[i]
        seq[start:end] = 'S'
        seq[end:] ='P'

    tags = tags.apply(lambda x: ''.join(x)) #put back together to string
    return tags


df = pd.read_csv('data/plasmodium_signal_peptides/uniprot_06072020.tsv', sep = '\t')

# Split 'Signal peptide'
df[['Signal peptide', 'Evidence']] = df['Signal peptide'].str.lstrip('SIGNAL').str.split(';', expand = True)[[0,1]]

#start and end idx to list
df['Signal peptide'] = df['Signal peptide'].str.split('\.\.')

# clean up evidence
df['Evidence'] = df['Evidence'].str.lstrip().str.replace('/evidence=', '').str.replace('"', '')

#some SPs have unknown positions with a ? --> useless
unknown_positions = df['Signal peptide'].apply(lambda x: '?' in x)
df = df[~unknown_positions]

#fix 1-0 indexing
df['Signal peptide'] = df['Signal peptide'].apply(lambda x: [int(i)-1 for i in x])

#make tags and save
df['Tags'] = make_sequence_tags(df['Sequence'], df['Signal peptide'])
df[['Sequence', 'Tags']].to_csv('data/plasmodium_signal_peptides/testout.tsv', sep='\t', index = False)
