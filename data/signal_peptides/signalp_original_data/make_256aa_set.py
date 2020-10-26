'''
Manually access Uniprot to download

1. Get accession numbers from training set file
grep '>' train_set.fasta | cut -c2- | cut -f1 -d "|" >accids.txt
2. Retrived id mapping and download fasta manually

'''

from typing import Union, Dict, Tuple
from pathlib import Path

def parse_fasta(filepath: Union[str, Path]) -> Dict[str, str]:

    seqs = {}
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            if line.startswith('>'):
                head = line.split('|')[1]
                seqs[head] = ''
            else:
                seqs[head] = seqs[head]+line

    return seqs


def parse_threeline_fasta(filepath: Union[str, Path]) -> Tuple[str,str,str]:

    with open(filepath, 'r') as f:
        lines = f.read().splitlines() #f.readlines()
        identifiers = lines[::3]
        sequences = lines[1::3]
        labels = lines[2::3]

    return identifiers, sequences, labels



if __name__ =='__main__':
    seqs = parse_fasta('full_length_train_set.fasta')
    ids, oldseqs, labels = parse_threeline_fasta('train_set.fasta')

    with open('train_set_256aa.fasta', 'w') as f:
        for idx, ident in enumerate(ids):
            acc = ident.lstrip('>').split('|')[0]
            f.write(ident+'\n')
            try:
                f.write(seqs[acc][:256] + '\n')
            except:
                print('Needs manual fix!')
                f.write('***Not in input data'+ oldseqs[idx]+'\n')
            f.write(labels[idx] + '\n')