import pandas as pd
import numpy as np

from shared_utils import parse_tm, get_kingdom, generate_label_sequence, string_to_structdf




if __name__ == '__main__':

    # Load prosite hits + non-hits. FN are covered by non-hits here (checked before)
    df = pd.read_csv('lipo/lipo_uniprot_prosite_matches.tsv', sep='\t')
    df2 = pd.read_csv('lipo/lipo_false_negatives.tsv', sep='\t')

    df = pd.concat([df,df2])

    #truncate sequences 
    df['Sequence'] = df['Sequence'].apply(lambda x: x[:70])


    # Remove unknowns
    df_false = pd.read_csv('lipo/lipo_unknown.tsv', sep='\t')

    len_before = len(df)
    df = df.merge(df_false['Entry name'].drop_duplicates(), on=['Entry name'], how='left', indicator=True)
    df = df[df['_merge'] == 'left_only']

    print(f'Removed unknowns. {len(df)} of {len_before} sequences remaining.')


    # Parse SPs
    sps = df['Signal peptide'].str.split(';', expand=True)[0]
    #convert ? to nan
    sp_ends = sps.apply(lambda x: x.split('..')[1] if type(x) == str else '?')
    sp_ends = sp_ends.apply(lambda x: int(x) if '?' not in x else np.nan)
    df['sp_end'] = sp_ends

    #if sp is nan, get info from scanprosite
    df_scanprosite = pd.read_csv('lipo/lipo_scanprosite_prosite_matches.tsv', sep='\t', names= ['id', 'match_start', 'match_end', 'profile', 'score', '5','6','7'])
    df_scanprosite['Entry'] = df_scanprosite['id'].str.split('|', expand=True)[1]
    df_scanprosite = df_scanprosite.set_index('Entry')

    for idx, row in df.iterrows():
        if np.isnan(row['sp_end']):
            try:
                df.loc[idx,'sp_end'] = df_scanprosite.loc[row['Entry']]['match_end']
            except KeyError:
                print(f"{row['Entry']} not in scanprosite result.removing from df.")

    df = df[~df['sp_end'].isna()]
    
    print(f'Removed ? CS. {len(df)} of {len_before} sequences remaining.')

    df = df[df['sp_end']<70]

    print(f'Removed long SPs. {len(df)} of {len_before} sequences remaining.')


    # Parse TMs
    df['tm_positions'] = parse_tm(df['Transmembrane'])
    



    # remove contradictions

    # tm positions are ordered, so taking the first start is enough
    # for contradictions, we also consider non-experimental assertions in uniprot
    tm_pos_1 = df['tm_positions'].apply(lambda x: x[0][0] if type(x) == list else x)
    df = df[~(tm_pos_1 <= df['sp_end'])]

    print(f'Removed contradictions. {len(df)} of {len_before} sequences remaining.')


    # remove virus and unknown taxonomies
    df['kingdom'] = df['Taxonomic lineage (PHYLUM)'].apply(lambda x: get_kingdom(x))
    df = df[df['kingdom'] != 'UNKNOWN']

    print(f'Removed bad taxonomies. {len(df)} of {len_before} sequences remaining.')


    #load TM data for label creation
    df_topdb = pd.read_csv('tm/tm_not_in_signalp5.tsv', sep='\t')
    df_topdb = df_topdb.set_index('acc')
    #filter for experimental codes in uniprot, use those when TOPDB doesn't have anything
    df['tm_positions'] = parse_tm(df['Transmembrane'],filter_exp=True)

    # Create labels

    with open('lipo.fasta', 'w') as f:

        for idx, row in df.iterrows():
            header = '>' + row['Entry'] + '|' + row['kingdom'] +  '|' + 'LIPO'

            try:
                tm_indices =  df_topdb.loc[row['Entry name']]
                struct = string_to_structdf(tm_indices['struct'])
                tm_indices = list(zip(list(struct['Begin']), list(struct['End']))) #list of start,end tuples
            except KeyError:
                #Resort to experimental uniprot annotations.
                tm_indices =row['tm_positions'] if not np.isnan(row['tm_positions']) else None

            label = generate_label_sequence(row['sp_end'], seq_len = len(row['Sequence']), tm_indices = tm_indices ,sp_symbol ='L' )

            f.write(header + '\n')
            f.write(row['Sequence'] + '\n')
            f.write(label + '\n')