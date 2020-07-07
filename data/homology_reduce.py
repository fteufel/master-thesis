from sklearn.model_selection import train_test_split
from Bio.SeqIO.FastaIO import SimpleFastaParser
import pandas as pd
import argparse
import logging
import sys
import os
import re
sys.path.append('../..')
sys.path.append('/zhome/1d/8/153438/experiments/master-thesis/')
from train_scripts.training_utils import VirtualBatchTruncatedBPTTHdf5Dataset

def fastaheader_to_dict(title):
    '''Parse fasta header as returned by mmseqs2
    '''
    base, acc, seqid, rest = title.split("|")
    matchstr = r'\s(\d{1,5})\s(?!.+\d)' #whitespace -number- whitespace, not followed by any number until the end of string
    try:
        organism, length, genus = re.split(matchstr, rest) 
    except ValueError: #some don't have genus info
        organism, length, genus = re.split(r'\s(\d{1,5})$', rest)
    organism = organism.rstrip().lstrip()
    length = int(length)
    genus = genus.rstrip().lstrip()
    return {'genus':genus, 'species':organism, 'length': length, 'accession_number': acc , 'name': seqid}



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default = 'data/dump_15062020/clustering_result/clusterRes_rep_seq.fasta',
                    help = 'Path to mmseqs clustering representative sequences fasta file')
parser.add_argument('--output_dir', type=str, default = 'homology_reduced_splits',
                    help = 'Path to save results')

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



#parse fasta
with open(args.data) as fasta_file:
    records = []
    for title, sequence in SimpleFastaParser(fasta_file):
        fields = fastaheader_to_dict(title)
        fields['Sequence'] = sequence
        records.append(fields) 

#to pandas
euk_df = pd.DataFrame.from_dict(records)
euk_df['Plasmodium'] = euk_df['genus'] =='Plasmodium' #make dummy indicator for stratified split

train_df, test_df = train_test_split(euk_df, stratify = euk_df["Plasmodium"], test_size = 0.3)
train_df, val_df = train_test_split(train_df, stratify = train_df["Plasmodium"], test_size = 0.1)

train_df_plasm = train_df.loc[train_df['Plasmodium']]
val_df_plasm= val_df.loc[val_df['Plasmodium']]
test_df_plasm= test_df.loc[test_df['Plasmodium']]


#save

eukarya_dir = os.path.join(args.output_dir, 'eukarya')
if not os.path.exists(eukarya_dir):
    os.mkdir(eukarya_dir)

train_df.to_csv(os.path.join(eukarya_dir, 'eukarya_train_full.tsv'), sep = '\t')
test_df.to_csv(os.path.join(eukarya_dir, 'eukarya_test_full.tsv'), sep = '\t')
val_df.to_csv(os.path.join(eukarya_dir, 'eukarya_val_full.tsv'), sep = '\t')


#Plasmodium only split
logger.info('Saving plasmodium only splits...')
plasmodium_dir = os.path.join(args.output_dir, 'plasmodium')
if not os.path.exists(plasmodium_dir):
    os.mkdir(plasmodium_dir)
train_df_plasm.to_csv(os.path.join(plasmodium_dir, 'plasmodium_train_full.tsv'), sep ='\t')
test_df_plasm.to_csv(os.path.join(plasmodium_dir, 'plasmodium_test_full.tsv'), sep ='\t')
val_df_plasm.to_csv(os.path.join(plasmodium_dir, 'plasmodium_val_full.tsv'), sep ='\t')


#For convenience - also save sequence only. Can be used by the dataset classes.
logger.info('Saving sequence .txt files')
train_df_plasm['Sequence'].to_csv(os.path.join(plasmodium_dir, 'train.txt'), header = None, index = None)
test_df_plasm['Sequence'].to_csv(os.path.join(plasmodium_dir, 'test.txt'), header = None, index = None)
val_df_plasm['Sequence'].to_csv(os.path.join(plasmodium_dir, 'valid.txt'), header = None, index = None)

train_df['Sequence'].to_csv(os.path.join(eukarya_dir, 'train.txt'), header = None, index = None)
test_df['Sequence'].to_csv(os.path.join(eukarya_dir, 'test.txt'), header = None, index = None)
val_df['Sequence'].to_csv(os.path.join(eukarya_dir, 'valid.txt'), header = None, index = None)

logger.info(f'Train seqs euk: {len(train_df)}')
logger.info(f'Test  seqs euk: {len(test_df)}')
logger.info(f'Valid seqs euk: {len(val_df)}')
logger.info(f'Train seqs pla: {len(train_df_plasm)}')
logger.info(f'Test  seqs pla: {len(test_df_plasm)}')
logger.info(f'Valid seqs pla: {len(val_df_plasm)}')
#
# Make Mdf5 files 
#
logger.info('Creating one-line .hdf5 files...')
dataset = VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_array(train_df['Sequence'], output_file=os.path.join(eukarya_dir, 'train.hdf5'))
logger.info(f'created {dataset.data_file}')
dataset = VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_array(test_df['Sequence'], output_file=os.path.join(eukarya_dir, 'test.hdf5'))
logger.info(f'created {dataset.data_file}')
dataset = VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_array(val_df['Sequence'], output_file=os.path.join(eukarya_dir, 'valid.hdf5'))
logger.info(f'created {dataset.data_file}')
dataset = VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_array(train_df_plasm['Sequence'], output_file=os.path.join(plasmodium_dir, 'train.hdf5'))
logger.info(f'created {dataset.data_file}')
dataset = VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_array(test_df_plasm['Sequence'], output_file=os.path.join(plasmodium_dir, 'test.hdf5'))
logger.info(f'created {dataset.data_file}')
dataset = VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_array(val_df_plasm['Sequence'], output_file=os.path.join(plasmodium_dir, 'valid.hdf5'))
logger.info(f'created {dataset.data_file}')