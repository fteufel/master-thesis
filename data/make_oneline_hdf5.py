import numpy as np
import pandas  as pd
import  argparse
import pandas as pd
import os
import sys
import logging
import pickle

sys.path.append('../..')
sys.path.append('/zhome/1d/8/153438/experiments/master-thesis/')
from train_scripts.training_utils import VirtualBatchTruncatedBPTTHdf5Dataset


parser = argparse.ArgumentParser(description='Balanced train-test-val split')
parser.add_argument('--data-dir', type=str, 
                    help='directory with the data')
args = parser.parse_args()

#make_hdf5_from_txt(cls, file: str, num_batches: int = 100, output_file: str = None, bptt_length = 75)


VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_txt(file = os.path.join(args.data_dir,'eukarya', 'val.txt'), 
                                                output_file= os.path.join(args.data_dir,'eukarya', 'valid.hdf5'))
VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_txt(file = os.path.join(args.data_dir,'eukarya', 'test.txt'), 
                                                output_file= os.path.join(args.data_dir,'eukarya', 'test.hdf5'))
VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_txt(file = os.path.join(args.data_dir,'eukarya', 'train.txt'), 
                                                output_file= os.path.join(args.data_dir,'eukarya', 'train.hdf5'))


VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_txt(file = os.path.join(args.data_dir,'plasmodium', 'valid.txt'), 
                                                output_file= os.path.join(args.data_dir,'plasmodium', 'valid.hdf5'))
VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_txt(file = os.path.join(args.data_dir,'plasmodium', 'test.txt'), 
                                                output_file= os.path.join(args.data_dir,'plasmodium', 'test.hdf5'))
VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_txt(file = os.path.join(args.data_dir,'plasmodium', 'train.txt'), 
                                                output_file= os.path.join(args.data_dir,'plasmodium', 'train.hdf5'))
