#Script to prepare data for AWD-LSTM
# save as hdf5, so that i can then do truncated backpropagation with indexing into the file.
#
import logging
from tape import TAPETokenizer, ProteinConfig
import sys
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/")
sys.path.append('..')
from train_scripts.training_utils import TruncatedBPTTHdf5Dataset
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Tokenize and batch ahead of model training')
parser.add_argument('--data', type=str, default='../data/awdlstmtestdata/',
                    help='location of the data corpus. Expects test, train and valid .txt')
parser.add_argument('--num_batches', type=int, default=200,
                    help='number of batches')
parser.add_argument('--output_file', type=str, default='data.hdf5',
                    help='number of batches')

args = parser.parse_args()


dataset = TruncatedBPTTHdf5Dataset.make_hdf5_from_txt(args.data,args.num_batches,args.output_file)

print('new hd5f file:')
print(dataset.data_file)