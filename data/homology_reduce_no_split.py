#homology reduce the clustering result of uniprot at 90% identity.
#make new data files with the same info as the previous ones, to run next clustering round on.
import pandas as pd
import argparse
import logging
import sys
import os
import re
sys.path.append('../..')
sys.path.append('/zhome/1d/8/153438/experiments/master-thesis/')
from train_scripts.training_utils import VirtualBatchTruncatedBPTTHdf5Dataset



parser = argparse.ArgumentParser(description='Homology reduce after mmseqs2 clustering')
parser.add_argument('--cluster-data', type=str, 
                    help='mmseqs2 clustering output .tsv')
parser.add_argument('--uniprot-data', type=str,
                    help = 'the uniprot table')
parser.add_argument('--output-dir', type=str,
                    help = 'dir to save outputs')
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S")
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

logger.addHandler(c_handler)
logger.addHandler(f_handler)



# load the data and sort
logger.info('loading UniProt table')
df_seqs = pd.read_csv(args.uniprot_data, sep = '\t')
#df_seqs = df_seqs.sort_values('Entry')

logger.info('loading clustering data')
df_cl = pd.read_csv(args.cluster_data, sep = '\t', header = None)
#df_cl = df_cl.sort_values(1)


#only retain reference sequences:

#1 set df_seqs index to Entry IDs
df_seqs = df_seqs.set_index('Entry')
#2 from clustering result, get the IDs of the clusters (= cluster name)
cluster_representatives = list(df_cl[0].unique())

#index cluster names into the df
df_reduced = df_seqs.loc[cluster_representatives, :]

#to pandas
df_reduced.to_csv(os.path.join(args.output_dir, 'reduced_dataset.tsv' ),sep ='\t')