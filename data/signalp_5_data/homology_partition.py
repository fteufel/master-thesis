'''
Homology partition a dataset from a distance matrix.

'''
import argparse
import pandas as pd
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--assignments', type = str, help ='graph-part output')
parser.add_argument('--fasta_file', type = str, help = 'Three-line format fasta file')
parser.add_argument('--output_dir', type = str)
args = parser.parse_args()

def parse_threeline_fasta(fasta_file):
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
        identifiers = lines[::3]
        sequences = lines[1::3]
        labels = lines[2::3]

    return pd.DataFrame.from_dict({'ID': identifiers, 'Sequence': sequences, 'Label': labels})

def write_threeline_fasta(df, fasta_file):
    with open(fasta_file, 'w') as fout:
        for idx, fields in df.iterrows():
            fout.write(fields['ID'])
            fout.write(fields['Sequence'])
            fout.write(fields['Label'])

#load dataset
seq_df = parse_threeline_fasta(args.fasta_file)
seq_df['AC'] = seq_df['ID'].apply(lambda x: x.split('|')[0].lstrip('>'))
#load distance matrix
assignments_df = pd.read_csv(args.assignments)


seq_df = seq_df.set_index('AC')
assignments_df = assignments_df.set_index('AC')

full_df = assignments_df.join(seq_df)

#save each partition as 3-line fasta
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

for idx in assignments_df['cluster'].unique():
    part_df = full_df.loc[full_df['cluster'] == idx]
    write_threeline_fasta(part_df, os.path.join(args.output_dir, f'partition_{idx}.fasta'))
#cluster
#partition clusters

