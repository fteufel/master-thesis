'''
Extract SPs from UniProt .tsv file.
Must have columns 'Sequence' and 'Signal peptide'
Saves extracted SPs as .fasta file for alignment.
SPs with unclear annotations will be dropped.
'''

import pandas as pd
import argparse
import os

def extract_sps(df):
    df[['SP', 'Evidence']] = df['Signal peptide'].str.lstrip('SIGNAL').str.split(';', expand = True)[[0,1]]
    df[['SP_start', 'SP_end']] = df['SP'].str.split('\.\.', expand=True)

    headers = []
    sps = []
    for idx, row in df.iterrows():

        if row['SP_end'] != '?':
            sp = row['Sequence'][:int(row['SP_end'])]
            header = '>'+ row['Entry']
            sps.append(sp)
            headers.append(header)
        else:
            print(f"Could not process {row['Entry']}")

    return headers, sps



if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='+', help='files to process')
    parser.add_argument('--output_dir', type=str, default = '.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for f in args.files:
        print(f'processing {f}')
        out_name = f.rstrip('.tsv') + '_sps_only.fasta'

        try:
            df = pd.read_csv(f, sep='\t')
        except:
            print('Failed!!!')
            import ipdb; ipdb.set_trace()
        headers, sps = extract_sps(df)

        with open(os.path.join(args.output_dir, out_name), 'w') as f:

            for header, sp in zip(headers, sps):
                f.write(header+'\n')
                f.write(sp + '\n')