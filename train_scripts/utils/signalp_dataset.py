'''
Dataloader to deal with the 3-line fasta format used in SignalP.
'''
import torch
from torch.utils.data import Dataset
from tape import TAPETokenizer
from typing import Union, List, Dict, Any, Sequence
import numpy as np
from pathlib import Path

SIGNALP_VOCAB = ['S', 'T', 'L', 'I', 'M', 'O']

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

def parse_threeline_fasta(filepath: Union[str, Path]):

    with open(filepath, 'r') as f:
        lines = f.read().splitlines() #f.readlines()
        identifiers = lines[::3]
        sequences = lines[1::3]
        labels = lines[2::3]

    return identifiers, sequences, labels

class SP_label_tokenizer():
    '''[S: Sec/SPI signal peptide | T: Tat/SPI signal peptide | L: Sec/SPII signal peptide | I: cytoplasm | M: transmembrane | O: extracellular]'''
    def __init__(self, labels: List[str] = SIGNALP_VOCAB):
        #build mapping
        token_ids = list(range(len(labels)))
        self.vocab  = dict(zip(labels, token_ids ))

    def tokenize(self, text: str) -> List[str]:
        return [x for x in text]

    def convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        try:
            return self.vocab[token]
        except KeyError:
            raise KeyError(f"Unrecognized token: '{token}'")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]

    def sequence_to_token_ids(self, sequence) -> List[int]:
        tokens = self.tokenize(sequence)
        ids = self.convert_tokens_to_ids(tokens)
        return ids


class ThreeLineFastaDataset(Dataset):
    """Creates a dataset from a SignalP format 3-line .fasta file.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 tokenizer: Union[str, TAPETokenizer] = 'iupac'
                 ):

        super().__init__()
        
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        self.label_tokenizer = SP_label_tokenizer()

        self.data_file = Path(data_path)
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)
        
        _, self.sequences, self.labels = parse_threeline_fasta(self.data_file)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index):
        item = self.sequences[index]
        labels = self.labels[index]
        token_ids = self.tokenizer.tokenize(item)# + [self.tokenizer.stop_token]
        token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)

        label_ids = self.label_tokenizer.sequence_to_token_ids(labels)
        #input_mask = np.ones_like(token_ids)

        return np.array(token_ids), np.array(label_ids)

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        #input_ids, input_mask, clan, family = tuple(zip(*batch))
        input_ids, label_ids = tuple(zip(*batch))
        data = torch.from_numpy(pad_sequences(input_ids, 0))
        # ignore_index is -1
        targets = torch.from_numpy(pad_sequences(label_ids, -1))

        return data, targets