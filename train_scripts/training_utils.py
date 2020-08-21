#Utils for handling data and other things for training loops
#Felix June 2020
#Most components adapted from TAPE repo

import torch
from torch.utils.data import Dataset
import typing
from typing import List, Tuple, Union, Any, Dict, Sequence
from pathlib import Path
import numpy as np
import pandas as pd
from tape import TAPETokenizer, ProteinConfig
import os
import wandb
import argparse
import logging
import h5py
import json


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
logger.addHandler(c_handler)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is None:
        return None
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

class TruncatedBPTTDataset(Dataset):
    """Creates a dataset from and enables truncated backpropagation through time training.
       Needs to load full data into memory, as it concatenates the full data to one sequence, and cuts it into batch_size pieces of equal length.
       __getitem__ then returns a seq_len (stochastic) sequence batch part, sequentially until end is reached

       File of n lines with sequences -> tensor [1, total_seq_len] -> tensor [seq_len/batch_size, batch_size]
       -> stochastic cutting of minibatches tensor [stochastic_seq_len, batch_size] until (seq_len/batch_size) is reached.

       Important: Needs to be used with a DataLoader with batch_size = 1, because the batch_size is already defined here.
       Seemed easier than also subclassing DataLoader to deal with that.
       It is also necessary to use the collate_fn, to get rid of the new dummy batch dimension added by the size=1 dataloader.
    
    Args:
        data_file (Union[str, Path]): Path to sequence file (line by line, just sequence nothing else).
    """
    def __init__(self,
                 data_file: Union[str, Path],
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 batch_size: int = 50,
                 bptt_length: int = 75
                 ):
        super().__init__()

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.bptt_length = bptt_length


        self.data_file = Path(data_file)
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(self.data_file)

        fn = Path(data_file+'.data')
        if os.path.exists(fn):
            logger.info('Loading cached dataset...')
            data = torch.load(fn)
        else:
            logger.info('Producing dataset...')
            data = self._concatenate_full_dataset(data_file)
            torch.save(data, fn)
            logger.info(f'Cached dataset at {fn}')

        data = self._batchify(data, self.batch_size)
        self.data = data
        self.start_idx, self.end_idx = self._get_bptt_indices(len(self.data), self.bptt_length)

        
    def _concatenate_full_dataset(self, data_file: Path) -> torch.LongTensor:
        '''
        tokenizes line-by-line dataset and concatenates to one long sequence of length (total_tokens)
        '''
        #with open(data_file, 'r') as f:
            #count the total length, needed to setup the tensor (really dumb, will change to list)
        #    tokens = 0
        #    for line in f:
        #        words = self.tokenizer.tokenize(line.rstrip()) + [self.tokenizer.stop_token]
        #        tokens += len(words)

        #tokenlist = [], tokenlist.append(token), LongTensor(tokenlist) saves first loop
        lines = 0
        tokenlist = []
        with open(data_file, 'r') as f:
            #ids = torch.LongTensor(tokens)
            #token = 0
            for line in f:
                words = self.tokenizer.tokenize(line.rstrip()) + [self.tokenizer.stop_token]
                #tokens += len(words)
                for word in words:
                    #ids[token] = self.tokenizer.convert_token_to_id(word)
                    tokenlist.append(self.tokenizer.convert_token_to_id(word))
                    #token += 1
                lines+=1
                if lines % 5000 ==0:
                    logger.info(f'processed {lines} lines.')
        return torch.LongTensor(tokenlist)

    def _batchify(self, data: torch.Tensor, bsz: int) -> torch.Tensor:
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data

    def _get_bptt_indices(self, total_length, bptt_length) -> Tuple[list, list]:
        '''
        At each forward pass, the length of the sequence we process is decided stochastically. For dataloaders, we need to do that ahead of time,
        so that __len__ and __getitem__(idx) work as expected. __getitem__ will return  (batch_size, bptt_length[idx]) idx times, until seq_len is exhausted.
        '''
        start_idx = []
        end_idx = []
        i = 0
        while i < total_length - 1 - 1: #first -1 because of 0 indexing, 2nd -1 because last token can only be target
            bptt = bptt_length if np.random.random() < 0.95 else bptt_length / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            seq_len = min(seq_len if seq_len else bptt_length, total_length - 1 - i)
            start_idx.append(i)
            end_idx.append(i + seq_len)
            i += seq_len

        return (start_idx, end_idx)


    def __len__(self) -> int:
        return len(self.start_idx)

    def __getitem__(self, index: int) -> Tuple:
        start, end = self.start_idx[index], self.end_idx[index]

        minibatch_data = self.data[start:end]
        target = self.data[start+1:end+1]#.view(-1) reshaping is done at the loss calculation
        #item = self.data[index]
        #'input_mask = np.ones_like(token_ids)

        return (minibatch_data, target)


    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        This collate_fn makes sure that DataLoader does not introduce a batch dimension, as we already have this from the data processing.
        To get desired behavior instatiate DataLoader with batch_size =1 
        '''
        data, targets = batch[0]
        
        return data, targets  # type: ignore



class LineByLineDataset(Dataset):
    """Creates a Language Modeling  Dataset.
    Does not support out-of-core loading, full dataset in memory
    Args:
        data_file (Union[str, Path]): Path to sequence file (line by line, just sequence nothing else).

        
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):
        super().__init__()
        
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        self.data_file = Path(data_path)
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)
        seqs = []
        with open(self.data_file, 'r') as f:
            for line in f:
                seqs.append(line.rstrip())
        self.data = seqs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item)
        #input_mask = np.ones_like(token_ids)

        return token_ids#, input_mask, item['clan'], item['family']

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        #input_ids, input_mask, clan, family = tuple(zip(*batch))
        input_ids = batch
        data = torch.from_numpy(pad_sequences(input_ids, 0))
        #input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        # ignore_index is -1
        targets = torch.from_numpy(pad_sequences(input_ids, -1))
        #clan = torch.LongTensor(clan)  # type: ignore
        #family = torch.LongTensor(family)  # type: ignore

        #this would be batch_first otherwise.
        return data.permute(1,0), targets.permute(1,0)



def override_from_wandb(wandb_config: wandb.wandb_config.Config, cli_config: argparse.Namespace, model_config: ProteinConfig):
    '''Override the values in the 2 local configs by those that are in wandb.config (either received from server, or set before with config.update)
        Enables hyperparameter search
    '''
    for key in wandb_config.keys():
        if key in dir(model_config):
            setattr(model_config, key,wandb_config[key])
        if key in dir(cli_config):
            setattr(cli_config, key, wandb_config[key])
        else:
            print(f'could not find {key} in local configs.')
    return cli_config, model_config


class TruncatedBPTTHdf5Dataset(Dataset):
    """Creates a dataset from and enables truncated backpropagation through time training.
      Batch_size needs to be specified beforehand, when creating the hdf5 file.
       __getitem__ then returns a seq_len (stochastic) sequence batch part, sequentially until end is reached

       File of n lines with sequences -> tensor [1, total_seq_len] -> tensor [seq_len/batch_size, batch_size]
       -> stochastic cutting of minibatches tensor [stochastic_seq_len, batch_size] until (seq_len/batch_size) is reached.

       Important: Needs to be used with a DataLoader with batch_size = 1, because the batch_size is already defined here.
       Seemed easier than also subclassing DataLoader to deal with that.
       It is also necessary to use the collate_fn, to get rid of the new dummy batch dimension added by the size=1 dataloader.
    
    Args:
        data_file (Union[str, Path]): Path to sequence file (line by line, just sequence nothing else).
    """
    def __init__(self,
                 data_file: Union[str, Path],
                 bptt_length: int = 75
                 ):
        super().__init__()


        self.bptt_length = bptt_length


        self.data_file = Path(data_file)
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(self.data_file)
        try:
            h5f = h5py.File(data_file, 'r') 
        except OSError:
            raise TypeError('Not an hdf5 file. If you want to instantiate from .txt, use make_hdf5_from_txt()')
        self.data = h5f['tokenized_sequences'] #np.array of size [ total_tokens/num_batches, num_batches]

        self.start_idx, self.end_idx = self._get_bptt_indices(len(self.data), self.bptt_length)

    def _get_bptt_indices(self, total_length, bptt_length) -> Tuple[list, list]:
        '''
        At each forward pass, the length of the sequence we process is decided stochastically. For dataloaders, we need to do that ahead of time,
        so that __len__ and __getitem__(idx) work as expected. __getitem__ will return  (batch_size, bptt_length[idx]) idx times, until seq_len is exhausted.
        '''
        start_idx = []
        end_idx = []
        i = 0
        while i < total_length - 1 - 1: #first -1 because of 0 indexing, 2nd -1 because last token can only be target
            bptt = bptt_length if np.random.random() < 0.95 else bptt_length / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            seq_len = min(seq_len if seq_len else bptt_length, total_length - 1 - i)
            start_idx.append(i)
            end_idx.append(i + seq_len)
            i += seq_len

        return (start_idx, end_idx)


    def __len__(self) -> int:
        return len(self.start_idx)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start, end = self.start_idx[index], self.end_idx[index]

        minibatch_data = self.data[start:end]
        target = self.data[start+1:end+1]#.view(-1) reshaping is done at the loss calculation

        return (torch.tensor(minibatch_data), torch.tensor(target))


    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        This collate_fn makes sure that DataLoader does not introduce a batch dimension, as we already have this from the data processing.
        To get desired behavior instatiate DataLoader with batch_size =1 
        '''
        data, targets = batch[0]
        
        return data, targets  # type: ignore

    @classmethod
    def make_hdf5_from_txt(cls, file: str, num_batches: int, output_file: str = None, bptt_length = 75):
        '''Tokenize sequences from a line-by-line txt file, concatenate and cut into num_batch sequences.
           Save as mdf5, and return Dataset with this mdf5 as source.
        '''
        if not os.path.exists(file):
            raise FileNotFoundError(file)
        tokenizer = TAPETokenizer(vocab = 'iupac') 
        #load and tokenize
        tokenlist = []
        with open(file, 'r') as f:
            #ids = torch.LongTensor(tokens)
            #token = 0
            for line in f:
                words = tokenizer.tokenize(line.rstrip()) + [tokenizer.stop_token]
                #tokens += len(words)
                for word in words:
                    tokenlist.append(tokenizer.convert_token_to_id(word))


        #split into batches
            tokensperbatch = len(tokenlist) // num_batches
            end = tokensperbatch*num_batches #trim
            tokenlist = tokenlist[0:end]
            data =  np.array(tokenlist)
            data = data.reshape(-1, num_batches)
        
        if not output_file:
            output_file = file + '.hdf5'

        with h5py.File(output_file, "w") as f:
            f.create_dataset('tokenized_sequences', data=data)

        return cls(output_file, bptt_length)

    @classmethod
    def make_hdf5_from_array(cls, array: Union[np.array, pd.Series], num_batches: int, output_file: str, bptt_length = 75):
        '''Tokenize sequences from a line-by-line txt file, concatenate and cut into num_batch sequences.
           Save as mdf5, and return Dataset with this mdf5 as source.
        '''

        tokenizer = TAPETokenizer(vocab = 'iupac') 
        #load and tokenize
        tokenlist = []
        for seq in array:
            words = tokenizer.tokenize(seq) + [tokenizer.stop_token]
            #tokens += len(words)
            for word in words:
                tokenlist.append(tokenizer.convert_token_to_id(word))

        #split into batches
        tokensperbatch = len(tokenlist) // num_batches
        end = tokensperbatch*num_batches #trim
        tokenlist = tokenlist[0:end]
        data =  np.array(tokenlist)
        data = data.reshape(-1, num_batches)

        with h5py.File(output_file, "w") as f:
            f.create_dataset('tokenized_sequences', data=data)

        return cls(output_file, bptt_length)


class VirtualBatchTruncatedBPTTHdf5Dataset(Dataset):
    """Creates a dataset from and enables truncated backpropagation through time training.
    Takes hdf5 file of concatenated and tokenized sequence as input, works with offsets into the hdf5 to emulate cutting into batches.
    __getitem__ then returns a seq_len (stochastic) sequence batch part, sequentially until end is reached.

       stochastic cutting of minibatches tensor [stochastic_seq_len, batch_size] until (seq_len/batch_size) is reached.

       Important: Needs to be used with a DataLoader with batch_size = 1, because the batch_size is already defined here.
       Seemed easier than also subclassing DataLoader to deal with that.
       It is also necessary to use the collate_fn, to get rid of the new dummy batch dimension added by the size=1 dataloader.
    
    Args:
        data_file (Union[str, Path]): Path to the hdf5 file
        buffer_size: buffer size in tokens (per batch), total memory buffer_size x batch_size. Hdf5 encoding is 8 byte per token.
                     Default value takes 40 Megabyte*batch_size memory.
    """
    def __init__(self,
                 data_file: Union[str, Path],
                 batch_size: int = 100,
                 bptt_length: int = 75,
                 buffer_size: int = 5000000,
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.bptt_length = bptt_length
        self.buffer_size = buffer_size

        self.data_file = Path(data_file)
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(self.data_file)
        #try:
        h5f = h5py.File(data_file, 'r') 
        #except OSError:
        #    raise TypeError('Not an hdf5 file. If you want to instantiate from .txt, use make_hdf5_from_txt()')
        self.data = h5f['tokenized_sequences'] #np.array of size [ total_tokens]

        #get batch offsets:
        tokens_per_batch = self.data.shape[0]//batch_size
        self.start_offsets = np.array([tokens_per_batch*i for i in range(batch_size)]) 
        self.start_idx, self.end_idx = self._get_bptt_indices(tokens_per_batch, self.bptt_length)

        #initialize buffer
        if self.buffer_size > 0:
            #buffer may not be bigger than tokens per batch.
            self.buffer_size = min(self.buffer_size, tokens_per_batch)
            self.buffer_start = 0
            self.buffer_end = self.buffer_size

            #buffer is array of batch_size x buffer_len
            self.buffer = np.array([self.data[start:end] for start,end in zip(self.start_offsets, self.start_offsets + self.buffer_size)])

    def _get_bptt_indices(self, tokens_per_batch, bptt_length) -> Tuple[list, list]:
        '''
        At each forward pass, the length of the sequence we process is decided stochastically. For dataloaders, we need to do that ahead of time,
        so that __len__ and __getitem__(idx) work as expected. __getitem__ will return  (batch_size, bptt_length[idx]) idx times, until seq_len is exhausted.
        '''
        start_idx = []
        end_idx = []
        i = 0
        while i < tokens_per_batch - 1 - 1: #first -1 because of 0 indexing, 2nd -1 because last token can only be target
            bptt = bptt_length if np.random.random() < 0.95 else bptt_length / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            seq_len = min(seq_len if seq_len else bptt_length, tokens_per_batch - 1 - i)
            start_idx.append(i)
            end_idx.append(i + seq_len)
            i += seq_len

        return (start_idx, end_idx)


    def __len__(self) -> int:
        return len(self.start_idx)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''get an item from the data,  by accessing the bptt_indices and offsetting with the batch start indices.
        Also handles buffer reloading when buffer end is reached.
        '''
        #NOTE for buffering: data is encoded in integers already. 1 Million positions take 8 mb memory. train euk has 9156936624 positions.
        #these indices start at 0 (e.g. correct for batch at position 0 only)
        start, end = self.start_idx[index], self.end_idx[index]
            #access cache
        if self.buffer_size > 0 and start >= self.buffer_start and end+1 <= self.buffer_end: #+1 for target
            #print(f'reading from buffer. Requested {start} to {end}')
            #most of this [0] seems unnecssary in retrospective, could have just kept working with start, end
            subsequence_starts = self.start_offsets[0] - self.buffer_start +start #buffer represents seq from buffer_start, but indexes at 0. correct with -
            subsequence_ends =  self.start_offsets[0] - self.buffer_start +end
            minibatch_data = self.buffer[:,subsequence_starts:subsequence_ends].transpose()
            target = self.buffer[:, subsequence_starts+1:subsequence_ends +1].transpose()

        else:
            print(f'Not reading from buffer. Requested {start} to {end}')
            #correct for other batches, add their offset into the sequence
            subsequence_starts= self.start_offsets + start
            subsequence_ends = self.start_offsets+end
            #index into the hdf5 array to extract all the subsequences
            #this makes batch_size dimension 0, need a reshape .T
            batchlist = [self.data[start:end] for start,end in zip(subsequence_starts, subsequence_ends)]
            minibatch_data = np.array(batchlist).transpose()
            batchlist = [self.data[start+1:end+1] for start,end in zip(subsequence_starts, subsequence_ends)]
            target = np.array(batchlist).transpose()

            if self.buffer_size > 0: #refresh buffer
                self.buffer_start = end
                # if there are not buffer_size tokens left in the last batch, need to reduce buffer size to what is left.
                if self.start_offsets[-1] +self.buffer_start + self.buffer_size > self.data.shape[0]:
                    self.buffer_size = self.data.shape[0] - self.buffer_start - self.start_offsets[-1]
                    print(f'Remaining dataset smaller than requested buffer. Adapted buffer size to {self.buffer_size}.')

                self.buffer_end = end + self.buffer_size

                self.buffer = np.array([self.data[start:end] for start,end in zip(self.start_offsets+self.buffer_start, self.start_offsets +self.buffer_start + self.buffer_size)])
                print(f'updated buffer. Now from {self.buffer_start} to {self.buffer_end}')
                assert self.buffer.shape[1] != 0
        return (torch.tensor(minibatch_data), torch.tensor(target))


    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        This collate_fn makes sure that DataLoader does not introduce a batch dimension, as we already have this from the data processing.
        To get desired behavior instatiate DataLoader with batch_size =1 
        '''
        data, targets = batch[0]
        
        return data, targets  # type: ignore

    @classmethod
    def make_hdf5_from_txt(cls, file: str, num_batches: int = 100, output_file: str = None, bptt_length = 75, buffer_size = 1000):
        '''Tokenize sequences from a line-by-line txt file, concatenate and cut into num_batch sequences.
           Save as mdf5, and return Dataset with this mdf5 as source.
        '''
        if not os.path.exists(file):
            raise FileNotFoundError(file)
        tokenizer = TAPETokenizer(vocab = 'iupac') 
        #load and tokenize        
        startidxlist = []
        tokenlist = []
        current_start_idx = 0

        with open(file, 'r') as f:
            for line in f:

                startidxlist.append(current_start_idx)
                words = tokenizer.tokenize(line.rstrip()) + [tokenizer.stop_token]
                for word in words:
                    tokenlist.append(tokenizer.convert_token_to_id(word))
                current_start_idx = len(tokenlist)


        data =  np.array(tokenlist)
        startidx = np.array(startidxlist)
        if not output_file:
            output_file = file + '.hdf5'

        with h5py.File(output_file, "w") as f:
            f.create_dataset('tokenized_sequences', data=data)
            f.create_dataset('starting_indices', data = startidx)

        return cls(output_file, num_batches, bptt_length, buffer_size)

    @classmethod
    def make_hdf5_from_array(cls, array: Union[np.array, pd.Series], output_file: str, num_batches: int =100 , bptt_length = 75):
        '''Tokenize sequences from a line-by-line txt file, concatenate and cut into num_batch sequences.
        Save as mdf5, and return Dataset with this mdf5 as source.
        Properties of mdf5 file:
            dataset tokenized_sequences: concatenation of all tokenized sequences (stop tokens inserted). 1D array of size total_n_tokens
            dataset starting_indices: starting index in tokenized_sequences of each sequence. 1D array of size n_sequences
        '''

        tokenizer = TAPETokenizer(vocab = 'iupac') 
        #load and tokenize
        startidxlist = []
        tokenlist = []
        current_start_idx = 0
        for seq in array:
            
            startidxlist.append(current_start_idx)
            words = tokenizer.tokenize(seq) + [tokenizer.stop_token]
            for word in words:
                tokenlist.append(tokenizer.convert_token_to_id(word))
            current_start_idx = len(tokenlist)

        data =  np.array(tokenlist)
        startidx = np.array(startidxlist)
        with h5py.File(output_file, "w") as f:
            f.create_dataset('tokenized_sequences', data=data)
            f.create_dataset('starting_indices', data = startidx)

        return cls(output_file, bptt_length)


def save_training_status(output_dir, epoch, global_step, num_epochs_no_improvement, stored_loss, learning_rate_steps):
    """Utility function to save the training loop status as json. To be used when restarting a run.
    Necessary as wall time does not permit a full training loop without restarting.
    """
    with open(os.path.join(output_dir, "training_loop_status.json"), "w") as f: #assume output dir already created before calling this here
        data = {
            'wandb_name': os.environ["WANDB_NAME"],
            'epoch': epoch, 
            'global_step': global_step, 
            'num_epochs_no_improvement': num_epochs_no_improvement, 
            'stored_loss': stored_loss, 
            'learning_rate_steps': learning_rate_steps
        }
        json.dump(data, f)

def load_training_status(output_dir):
    """Utility to load training loop status from json.
    Expects training_loop_status.json in output_dir.
    """
    with open(os.path.join(output_dir, "training_loop_status.json"), "r") as f: 
        data = json.load(f)
        return data


#TODO refactor all Hdf5 Datasets, make them inherit the static methods from Hdf5Dataset base class

class Hdf5Dataset(Dataset):
    '''Base Class for Hdf5 Datasets. Implements the static factory methods to make Hdf5 files.
    '''
    def __init__(self):
        super().__init__()

    @classmethod
    def make_hdf5_from_txt(cls, file: str, num_batches: int = 100, output_file: str = None, bptt_length = 75, buffer_size = 1000):
        '''Tokenize sequences from a line-by-line txt file, concatenate and cut into num_batch sequences.
           Save as mdf5, and return Dataset with this mdf5 as source.
        '''
        if not os.path.exists(file):
            raise FileNotFoundError(file)
        tokenizer = TAPETokenizer(vocab = 'iupac') 
        #load and tokenize        
        startidxlist = []
        tokenlist = []
        current_start_idx = 0

        with open(file, 'r') as f:
            for line in f:

                startidxlist.append(current_start_idx)
                words = tokenizer.tokenize(line.rstrip()) + [tokenizer.stop_token]
                for word in words:
                    tokenlist.append(tokenizer.convert_token_to_id(word))
                current_start_idx = len(tokenlist)


        data =  np.array(tokenlist)
        startidx = np.array(startidxlist)
        if not output_file:
            output_file = file + '.hdf5'

        with h5py.File(output_file, "w") as f:
            f.create_dataset('tokenized_sequences', data=data)
            f.create_dataset('starting_indices', data = startidx)

        return cls(output_file, num_batches, bptt_length, buffer_size)

    @classmethod
    def make_hdf5_from_array(cls, array: Union[np.array, pd.Series], output_file: str, num_batches: int =100 , bptt_length = 75):
        '''Tokenize sequences from a line-by-line txt file, concatenate and cut into num_batch sequences.
        Save as mdf5, and return Dataset with this mdf5 as source.
        Properties of mdf5 file:
            dataset tokenized_sequences: concatenation of all tokenized sequences (stop tokens inserted). 1D array of size total_n_tokens
            dataset starting_indices: starting index in tokenized_sequences of each sequence. 1D array of size n_sequences
        '''

        tokenizer = TAPETokenizer(vocab = 'iupac') 
        #load and tokenize
        startidxlist = []
        tokenlist = []
        current_start_idx = 0
        for seq in array:
            
            startidxlist.append(current_start_idx)
            words = tokenizer.tokenize(seq) + [tokenizer.stop_token]
            for word in words:
                tokenlist.append(tokenizer.convert_token_to_id(word))
            current_start_idx = len(tokenlist)

        data =  np.array(tokenlist)
        startidx = np.array(startidxlist)
        with h5py.File(output_file, "w") as f:
            f.create_dataset('tokenized_sequences', data=data)
            f.create_dataset('starting_indices', data = startidx)

        return cls(output_file, bptt_length)

class FullSeqHdf5Dataset(Hdf5Dataset):
    """Creates a dataset from Hdf5. Loads full sequences by using the index array of the .hdf5 file to index into the concatenated array.
    No buffer implemented at the moment. buffer_size is not used.
    Args:
        data_file:   Path to the hdf5 file
        buffer_size: buffer size in tokens (per batch), total memory buffer_size x batch_size. Hdf5 encoding is 8 byte per token.
                     Default value takes 40 Megabyte memory.
    """
    def __init__(self,
                 data_file: Union[str, Path],
                 buffer_size: int = 5000000,
                 ):
        super().__init__()

        self.buffer_size = buffer_size

        self.data_file = Path(data_file)
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(self.data_file)
        #try:
        h5f = h5py.File(data_file, 'r') 
        #except OSError:
        #    raise TypeError('Not an hdf5 file. If you want to instantiate from .txt, use make_hdf5_from_txt()')
        self.data = h5f['tokenized_sequences'] #np.array of size [ total_tokens]
        self.indices = h5f['starting_indices'] #np.array of size[total_n_seqs]

        #initialize buffer
        if self.buffer_size > 0:
            #buffer may not be bigger than tokens per batch.
            #self.buffer_size = min(self.buffer_size, tokens_per_batch)
            self.buffer_start = 0
            self.buffer_end = self.buffer_size
            self.buffer = self.data[self.buffer_start:self.buffer_end]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''get an item from the data,  by accessing the bptt_indices and offsetting with the batch start indices
        '''
        #NOTE for buffering: data is encoded in integers already. 1 Million positions take 8 mb memory. train euk has 9156936624 positions.
        start = self.indices[index] #last position: take until end, not until next start token
        end  =  self.indices[index+1] if index < len(self.indices) -1 else len(self.data)
        #access cache
        
        #if self.buffer_size > 0 and start >= self.buffer_start and end <= self.buffer_end:
           
        #    data = self.buffer[start:end]
        #else:
        #    data = self.data[start:end]
        #
        #    if self.buffer_size > 0: #refresh buffer TODO disabled because buffer refreshing is broken
        #        self.buffer_start = end
        #        # if there are not buffer_size tokens left in the last batch, need to reduce buffer size to what is left.
        #        if self.buffer_start + self.buffer_size > self.data.shape[0]:
        #            self.buffer_size = self.data.shape[0] - self.buffer_start
        #            print(f'Remaining dataset smaller than requested buffer. Adapted buffer size to {self.buffer_size}.')
        #
        #        self.buffer_end = end + self.buffer_size
        #        self.buffer = self.data[self.buffer_start:self.buffer_end]
        #        print(f'updated buffer. Now from {self.buffer_start} to {self.buffer_end}')
        
        data = self.data[start:end]
        #make target from data
        target = np.append(data[1:], -1) # output at last token is ignored, is the stop token. nothing next to predict.
        return data, target

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        This collate_fn makes sure that DataLoader does not introduce a batch dimension, as we already have this from the data processing.
        To get desired behavior instatiate DataLoader with batch_size =1 
        '''
        data, targets = tuple(zip(*batch))
        
        torch_data = torch.from_numpy(pad_sequences(data, 0)) #0 is tokenizer pad token
        torch_targets = torch.from_numpy(pad_sequences(targets, -1)) #pad with -1 to ignore loss

        return torch_data.permute(1,0), torch_targets.permute(1,0)  # type: ignore



class SequenceTaggingDataset(Dataset):
    """Creates a Sequence Tagging Dataset
    Does not support out-of-core loading, full dataset in memory.
    Returns batch_first
    Args:
        data_file (Union[str, Path]): Path to tab-separated input file.
                Column 1: Sequence, Column 2: Tags. Need to be of same length always
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 tokenizer: Union[str, TAPETokenizer] = 'iupac', label_dict = None):
        super().__init__()
        
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        if not os.path.exists(data_file):
            raise FileNotFoundError(data_file)
        
        if label_dict is None:
            self.label_dict = {'S': 0,'P': 1} #TODO make arg or learn from input data once
        else:
            self.label_dict = label_dict
        
        df = pd.read_csv(data_file, sep ='\t')
        self.data = df[df.columns[0]]
        self.labels = df[df.columns[1]]
        #no more pandas from here
        self.data = list(self.data)
        self.labels = list(self.labels)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        labels = self.labels[index]
        #don't use default tokenizer.encode, want control over special tokens here.
        tokens = self.tokenizer.tokenize(item) #TODO remove special tokens here
        token_ids = np.array(self.tokenizer.convert_tokens_to_ids(tokens))
        label_ids = self._encode_labels(labels)
        input_mask = np.ones_like(token_ids)
        assert len(token_ids) == len(label_ids), 'token length and label length are not the same!'
        return token_ids, label_ids, input_mask
    
    def _encode_labels(self,labelstring):
        out = []
        for pos in labelstring:
            out.append(self.label_dict[pos])
        return np.array(out)

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, label_ids, input_mask =  tuple(zip(*batch))
        data = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        targets = torch.from_numpy(pad_sequences(label_ids, -1))         # ignore_index is -1
        #this would be batch_first otherwise.
        return data, targets, input_mask#data.permute(1,0), targets.permute(1,0), input_mask.permute(1,0)


class SequenceClassificationDataset(Dataset):
    '''multi-class classification of sequences.
    Expects Uniprot tab delimited file. Columns to use are specified as args.
    '''

    def __init__(self,
                 data_file: Union[str, Path],
                 tokenizer: Union[str, TAPETokenizer] = 'iupac', label_column = 'target label', sequence_column = 'Sequence'):
        super().__init__()
        
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        if not os.path.exists(data_file):
            raise FileNotFoundError(data_file)
        
        
        df = pd.read_csv(data_file, sep ='\t')
        self.data = df[sequence_column]
        self.labels = df[label_column]
        #no more pandas from here
        self.data = list(self.data)
        self.labels = list(self.labels)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        label = self.labels[index]

        tokens = self.tokenizer.tokenize(item)
        token_ids = np.array(self.tokenizer.convert_tokens_to_ids(tokens))
        input_mask = np.ones_like(token_ids)
        return token_ids, label, input_mask

    def collate_fn(self, batch: List[Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids, label_ids, input_mask =  tuple(zip(*batch))
        data = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        targets = torch.tensor(label_ids)

        return data, targets, input_mask