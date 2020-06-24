#Utils for handling data and other things for training loops
#Felix June 2020
#Most components adapted from TAPE repo

import torch
from torch.utils.data import Dataset
import typing
from typing import List, Tuple, Union, Any, Dict, Sequence
from pathlib import Path
import numpy as np
from tape import TAPETokenizer, ProteinConfig

import wandb
import argparse


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
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        data = self._concatenate_full_dataset(data_file)
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