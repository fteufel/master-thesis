'''
We want to perform global alignment between a SP sequence of interest and all the SPs in the 
SignalP 5.0 training set.

Preprocess the traning data accordingly.
This script takes all positive samples from dataset and cuts the part of the sequence
that forms the SP.


'''

from typing import Union, Dict, Tuple
from pathlib import Path
import argparse

def parse_threeline_fasta(filepath: Union[str, Path]) -> Tuple[str,str,str]:

    with open(filepath, 'r') as f:
        lines = f.read().splitlines() #f.readlines()
        identifiers = lines[::3]
        sequences = lines[1::3]
        labels = lines[2::3]

    return identifiers, sequences, labels


def get_sp_end(labels):
    sp_ends = []
    '''find the position of the last S, T or L'''

    for label in labels:
        label = label.replace('T','S')
        label = label.replace('L','S')
        label = label.replace('P','S')
        sp_end = label.rfind('S') +1
        sp_ends.append(sp_end)

    return sp_ends


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='train_set.fasta')
    parser.add_argument('--output_file', type=str, default='train_set_for_alignment.fasta')
    args = parser.parse_args()

    ids, seqs, labels = parse_threeline_fasta(args.input_file)
    sp_ends = get_sp_end(labels)

    with open(args.output_file, 'w') as f:
        for idx, ident in enumerate(ids):
            typ = ident.lstrip('>').split('|')[2]
            #>P0A921|NEGATIVE|SP|4

            if typ != 'NO_SP':
                f.write(ident+'\n')
                sp_seq =  seqs[idx][:sp_ends[idx]]
                f.write(sp_seq + '\n')
