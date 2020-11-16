import pandas as pd
import numpy as np

from shared_utils import parse_tm, get_kingdom, generate_label_sequence, string_to_structdf


if __name__ == '__main__':

    # Load prosite hits + non-hits. FN are covered by non-hits here (checked before)

    # use original index to map to topdb df, query that generated it was 1:1 but names have changed since topdb release
    df = pd.read_csv('tm/tm_not_in_signalp_uniprot.tsv', sep='\t').reset_index()

    len_before = len(df)
    #truncate sequences 
    # Remove sequences that are no longer in Uniprot
    df = df[~df['Sequence'].isna()]
    print(f'Removed outdated sequences. {len(df)} of {len_before} sequences remaining.')
    df['Sequence'] = df['Sequence'].apply(lambda x: x[:70])


    # Remove SP, TAT and LIPO sps.
    df1 = pd.read_csv('lipo/lipo_uniprot_prosite_matches.tsv', sep='\t')
    df2 = pd.read_csv('lipo/lipo_false_negatives.tsv', sep='\t')
    df3 = pd.read_csv('tat/tat_uniprot_prosite_matches.tsv', sep='\t')
    df4 = pd.read_csv('tat/tat_uniprot_non_prosite_matches.tsv', sep='\t')
    df5 = pd.read_csv('sp/sp_uniprot_hits.tsv', sep='\t')
    df_false = pd.concat([df1, df2, df3, df4, df5])

    df = df.merge(df_false['Entry name'].drop_duplicates(), on=['Entry name'], how='left', indicator=True)
    df = df[df['_merge'] == 'left_only']

    print(f'Removed SP/TAT/LIPO hits. {len(df)} of {len_before} sequences remaining.')



    # remove virus and unknown taxonomies
    df['kingdom'] = df['Taxonomic lineage (PHYLUM)'].apply(lambda x: get_kingdom(x))
    df = df[df['kingdom'] != 'UNKNOWN']

    print(f'Removed bad taxonomies. {len(df)} of {len_before} sequences remaining.')


    #load TM data for label creation
    df_topdb = pd.read_csv('tm/tm_not_in_signalp5.tsv', sep='\t')
    #df_topdb = df_topdb.set_index('acc')

    # Create labels

    with open('tm.fasta', 'w') as f:

        for idx, row in df.iterrows():
            header = row['Entry'] + '|' + row['kingdom'] +  '|' + 'NO_SP'

            try:
                tm_indices =  df_topdb.loc[row['index']]
                struct = string_to_structdf(tm_indices['struct'])
                tm_indices = list(zip(list(struct['Begin']), list(struct['End']))) #list of start,end tuples
            except KeyError:
                print(f"Name mismatch {row['Entry name']}. Discard.")
                continue

            if tm_indices[0][0] > 70:
                continue

            label = generate_label_sequence(0, seq_len = len(row['Sequence']), tm_indices = tm_indices ,sp_symbol ='X' )

            f.write(header + '\n')
            f.write(row['Sequence'] + '\n')
            f.write(label + '\n')