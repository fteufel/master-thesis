import pandas as pd
import numpy as np

from shared_utils import parse_tm, get_kingdom, generate_label_sequence, string_to_structdf




if __name__ == '__main__':

    # Load prosite hits + non-hits. FN are covered by non-hits here (checked before)
    df = pd.read_csv('soluble/soluble_uniprot_hits.tsv', sep='\t')
    len_before = len(df)


    #truncate sequences 
    df['Sequence'] = df['Sequence'].apply(lambda x: x[:70])


    # Remove unknowns
    df1 = pd.read_csv('lipo/lipo_uniprot_prosite_matches.tsv', sep='\t')
    df2 = pd.read_csv('lipo/lipo_false_negatives.tsv', sep='\t')
    df3 = pd.read_csv('tat/tat_uniprot_prosite_matches.tsv', sep='\t')
    df4 = pd.read_csv('tat/tat_uniprot_non_prosite_matches.tsv', sep='\t')
    df5 = pd.read_csv('sp/sp_uniprot_hits.tsv', sep='\t')
    df6 = pd.read_csv('tm/tm_not_in_signalp_uniprot.tsv', sep='\t')
    df_false = pd.concat([df1, df2, df3, df4, df5, df6])

    df = df.merge(df_false['Entry name'].drop_duplicates(), on=['Entry name'], how='left', indicator=True)
    df = df[df['_merge'] == 'left_only']

    print(f'Removed SP/TAT/LIPO/TM hits. {len(df)} of {len_before} sequences remaining.')


    # Remove TM
    tm = df['Transmembrane'].apply(lambda x: type(x) == str)
    df = df[~tm]
    print(f'Removed TMs. {len(df)} of {len_before} sequences remaining.')


    # Remove fragments
    df =df[df['Fragment'].isna()]
    print(f'Removed fragments. {len(df)} of {len_before} sequences remaining.')

    # Remove short seqs- many peptides in set.
    df= df[~(df['Sequence'].str.len()<70)]
    print(f'Removed short seqs. {len(df)} of {len_before} sequences remaining.')


    # remove virus and unknown taxonomies
    df['kingdom'] = df['Taxonomic lineage (PHYLUM)'].apply(lambda x: get_kingdom(x))
    df = df[df['kingdom'] != 'UNKNOWN']

    print(f'Removed bad taxonomies. {len(df)} of {len_before} sequences remaining.')


    # Create labels

    with open('soluble.fasta', 'w') as f:

        for idx, row in df.iterrows():
            header = '>' + row['Entry'] + '|' + row['kingdom'] +  '|' + 'NO_SP'


            label = ''.join( ['I' * 70])

            f.write(header + '\n')
            f.write(row['Sequence'] + '\n')
            f.write(label + '\n')