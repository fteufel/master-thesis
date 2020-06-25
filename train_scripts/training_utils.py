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


#debug with this, print doesn't give outputs on hpc
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


###fully virtual reshaping from a concatenated hdf5 file
#tokens_per_batch = concatenated.shape[0]//num_batches
#start_offsets = [tokens_per_batch*i for i in range(num_batches)]  
#minibatch_end_offsets = start_offsets + 75

# indexes = np.array([np.arange(start,end) for start,end in zip(start_offsets, end1_offsets)])
#concatenated[indexes]

#Untested
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
    """
    def __init__(self,
                 data_file: Union[str, Path],
                 batch_size: int = 100,
                 bptt_length: int = 75,
                 buffer_size: int = 5, #how many batches to load into ram at once
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.bptt_length = bptt_length
        self.buffer_size = buffer_size

        self.data_file = Path(data_file)
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(self.data_file)
        try:
            h5f = h5py.File(data_file, 'r') 
        except OSError:
            raise TypeError('Not an hdf5 file. If you want to instantiate from .txt, use make_hdf5_from_txt()')
        self.data = h5f['tokenized_sequences'] #np.array of size [ total_tokens]

        #get batch offsets:
        tokens_per_batch = self.data.shape[0]//batch_size
        self.start_offsets = np.array([tokens_per_batch*i for i in range(batch_size)]) 
        self.start_idx, self.end_idx = self._get_bptt_indices(tokens_per_batch, self.bptt_length)

        #initialize buffer
        #if self.buffer_size > 0:
        #    self.buffer_indices = list(range(buffer_size))
        #    self.buffer = [self[x] for x in self.buffer_indices]

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
        '''get an item from the data,  by accessing the bptt_indices and offsetting with the batch start indices
        '''
        #NOTE buffering like this does not give me a performance boost. I would need to cut longer contiguous subsequences at a time, to then cut the bptts from that.
        #if  self.buffer_size > 0 and index in self.buffer_indices:
        #    i%self.buffer_size
        #    return self.buffer[i]
        #else:
        #   __getitem__ block
        #   if self.buffer_size > 0:
        #       self.buffer_indices = list(range(index+1, index+1+buffer_size))
        #       self.buffer = [self[x] for x in self.buffer_indices]
        #these indices start at 0 (e.g. correct for batch at position 0 only)
        start, end = self.start_idx[index], self.end_idx[index]
        #correct for other batches, add their offset into the sequence
        subsequence_starts= self.start_offsets + start
        subsequence_ends = self.start_offsets+end
        #make index into the hdf5 array to extract all the subsequences
        #indexes = np.array([np.arange(start,end) for start,end in zip(subsequence_starts, subsequence_ends)]) 
        #minibatch_data = self.data[indexes] #not supported by hdf5
        #this makes batch_size dimension 0, need a reshape .T
        batchlist = [self.data[start:end] for start,end in zip(subsequence_starts, subsequence_ends)]
        minibatch_data = np.array(batchlist)
        batchlist = [self.data[start+1:end+1] for start,end in zip(subsequence_starts, subsequence_ends)]
        target = np.array(batchlist)

        return (torch.tensor(minibatch_data).T, torch.tensor(target).T)


    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        This collate_fn makes sure that DataLoader does not introduce a batch dimension, as we already have this from the data processing.
        To get desired behavior instatiate DataLoader with batch_size =1 
        '''
        data, targets = batch[0]
        
        return data, targets  # type: ignore

    @classmethod
    def make_hdf5_from_txt(cls, file: str, num_batches: int = 100, output_file: str = None, bptt_length = 75):
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

        data =  np.array(tokenlist)

        if not output_file:
            output_file = file + '.hdf5'

        with h5py.File(output_file, "w") as f:
            f.create_dataset('tokenized_sequences', data=data)

        return cls(output_file, bptt_length)

    @classmethod
    def make_hdf5_from_array(cls, array: Union[np.array, pd.Series], output_file: str, num_batches: int =100 , bptt_length = 75):
        '''Tokenize sequences from a line-by-line txt file, concatenate and cut into num_batch sequences.
           Save as mdf5, and return Dataset with this mdf5 as source.
        '''

        tokenizer = TAPETokenizer(vocab = 'iupac') 
        #load and tokenize
        tokenlist = []
        for seq in array:
            
            words = tokenizer.tokenize(seq) + [tokenizer.stop_token]
            for word in words:
                tokenlist.append(tokenizer.convert_token_to_id(word))

        data =  np.array(tokenlist)
        with h5py.File(output_file, "w") as f:
            f.create_dataset('tokenized_sequences', data=data)

        return cls(output_file, bptt_length)