import torch
from torch.utils.data import Dataset
import typing
from pathlib import Path
import numpy as np
from tape import TAPETokenizer

class TruncatedBPTTDataset(Dataset):
    """Creates a dataset from and enables truncated backpropagation through time training.
       Needs to load full data into memory, as it concatenates the full data to one sequence, and cuts it into batch_size pieces of equal length.
       __getitem__ then returns a seq_len (stochastic) sequence batch part, sequentially until end is reached

       File of n lines with sequences -> tensor [1, total_seq_len] -> tensor [seq_len/batch_size, batch_size]
       -> stochastic cutting of minibatches tensor [stochastic_seq_len, batch_size] until (seq_len/batch_size) is reached.

       Important: Needs to be used with a DataLoader with batch_size = 1, because the batch_size is already defined here.
       Seemed easier than also subclassing DataLoader to deal with that.
       Also, dataloader will then add a batch_size dimension so that (1, bptt_length, batch_size)
       --> Better not use DataLoaders with this thing here. Iterate over Dataset directly.
       Or, create a collate_fn that unsqueezes.
    
    Args:
        data_file (Union[str, Path]): Path to sequence file (line by line, just sequence nothing else).
    """
    def __init__(self,
                 data_file: typing.Union[str, Path],
                 tokenizer: typing.Union[str, TAPETokenizer] = 'iupac',
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
        with open(data_file, 'r') as f:
            #count the total length, needed to setup the tensor (really dumb, will change to list)
            tokens = 0
            for line in f:
                words = self.tokenizer.tokenize(line.rstrip()) + [self.tokenizer.stop_token]
                tokens += len(words)
        with open(data_file, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = self.tokenizer.tokenize(line.rstrip()) + [self.tokenizer.stop_token]
                tokens += len(words)
                for word in words:
                    ids[token] = self.tokenizer.convert_token_to_id(word)
                    token += 1
        return ids

    def _batchify(self, data: torch.Tensor, bsz: int) -> torch.Tensor:
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data

    def _get_bptt_indices(self, total_length, bptt_length) -> typing.Tuple[list, list]:
        '''
        At each forward pass, the length of the sequence we process is decided stochastically. For dataloaders, we need to do that ahead of time,
        so that __len__ and __getitem__(idx) work as expected. __getitem__ will return  (batch_size, bptt_length[idx]) idx times, until seq_len is exhausted.
        '''
        start_idx = []
        end_idx = []
        i = 0
        while i < total_length - 1 - 1:
            bptt = bptt_length if np.random.random() < 0.95 else bptt_length / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            seq_len = min(seq_len if seq_len else bptt_length, total_length - 1 - i)
            start_idx.append(i)
            end_idx.append(i + seq_len)
            i += seq_len

        return (start_idx, end_idx)


    #len is wrong, because len (in terms of how many minibatches) is dynamic
    def __len__(self) -> int:
        return len(self.start_idx)

    def __getitem__(self, index):
        start, end = self.start_idx[index], self.end_idx[index]

        minibatch_data = self.data[start:end]
        target = self.data[start+1:end+1]#.view(-1) reshaping is done at the loss calculation
        #item = self.data[index]
        #'input_mask = np.ones_like(token_ids)

        return minibatch_data, target
