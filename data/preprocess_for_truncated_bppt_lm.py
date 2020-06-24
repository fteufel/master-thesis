#Script to prepare data for AWD-LSTM style truncated backpropagation through time language modeling ahead of time.
#
#
import logging
from tape import TAPETokenizer, ProteinConfig
import argparse




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
            ids = torch.LongTensor(tokens) #tokenlist = [], tokenlist.append(token), LongTensor(tokenlist) saves first loop
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