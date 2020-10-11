# Create a training corpus from UniRef100 data.
# whitespace between amino acids
# empty line between sequences
# uniprot:(taxonomy:"Archaea [2157]") AND identity:1.0

# random subset of UniRef100 w/o Archaea
# https://www.uniprot.org/uniref/?query=NOT%20uniprot%3A(taxonomy%3A%22Archaea%20%5B2157%5D%22)%20AND%20identity%3A1.0
# &columns=id%2Cname%2Csequence%2Ccommon%20taxon%2Ccommontaxonid
# &limit=3751449&format=tab

import re
import argparse
import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize()


parser =argparse.ArgumentParser('convert .tsv to huggingface .txt')
parser.add_argument('--data_file')
parser.add_argument('--output_file')
parser.add_argument('--data_file_2', default = None, help='optional 2nd file in same format to mix with data_file')
parser.add_argument('--split_by_length', default=None, help='split into two files based on sequence length')
parser.add_argument('--remove_max_length', default=None, help='drop sequences that are longer than this')
args = parser.parse_args()

sequences_Example = ["A E T C Z A O","S K T Z P"]
sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

df = pd.read_csv(args.data_file, sep='\t')

df['Reference sequence'] = df['Reference sequence'].parallel_apply(lambda x: " ".join(x)) #insert whitespace
df['Reference sequence'] = df['Reference sequence'].parallel_apply(lambda x: re.sub(r"[UZOB]", "X", x)) #replace rare AAs


if args.data_file_2 is not None:
    df_2 = pd.read_csv(args.data_file, sep='\t')

    df_2['Reference sequence'] = df_2['Reference sequence'].parallel_apply(lambda x: " ".join(x)) #insert whitespace
    df_2['Reference sequence'] = df_2['Reference sequence'].parallel_apply(lambda x: re.sub(r"[UZOB]", "X", x)) #replace rare AAs

    #concatenate and shuffle
    df = pd.concat([df, df_2])
    df = df.sample(frac=1) 

#ProtBert was first trained for 300k steps on sequences with a maximum length of 512 
# and then for another 100k steps on sequences with a length of a maximum length of 2k.
if args.split_by_length is not None:
    split_indicator = df['Reference sequence'].parallel_apply(lambda x: True if len(x)<=512 else False)
    df_short =df.loc[split_indicator]
    df_long = df.loc[~split_indicator]

    df_short.to_csv(args.output_file+'_short.txt', columns = ['Reference sequence'], index=False, header=False, line_terminator='\n')
    df_long.to_csv(args.output_file+'_long.txt', columns = ['Reference sequence'], index=False, header=False, line_terminator='\n')
else:
    #write to txt with empty lines in between
    df.to_csv(args.output_file+'.txt', columns = ['Reference sequence'], index=False, header=False, line_terminator='\n')
