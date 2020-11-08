'''
Dataset to deal with the 3-line fasta format used in SignalP.
'''
import torch
from torch.utils.data import Dataset
from typing import Union, List, Dict, Any, Sequence
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import PreTrainedTokenizer
from collections import defaultdict

from .signalp_label_processing import process_SP
#[S: Sec/SPI signal peptide | T: Tat/SPI signal peptide | L: Sec/SPII signal peptide | I: cytoplasm | M: transmembrane | O: extracellular]
SIGNALP_VOCAB = ['S', 'I' , 'M', 'O', 'T', 'L'] #NOTE eukarya only uses {'I', 'M', 'O', 'S'}
SIGNALP_GLOBAL_LABEL_DICT = {'NO_SP':0, 'SP':1,'LIPO':2, 'TAT':3}
SIGNALP_KINGDOM_DICT = {'EUKARYA': 0, 'POSITIVE':1, 'NEGATIVE':2, 'ARCHAEA':3}
SIGNALP6_GLOBAL_LABEL_DICT = {'NO_SP':0, 'SP':1,'LIPO':2, 'TAT':3, 'LIPOTAT':4, 'PILIN':5}

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

def subset_dataset(identifiers: List[str], sequences: List[str], labels: List[str], partition_id: List[int], kingdom_id: List[str], type_id: List[str]):
    '''Extract a subset from the complete .fasta dataset'''

    #break up the identifier into elements
    parsed = [ element.lstrip('>').split('|') for element in identifiers]
    acc_ids, kingdom_ids, type_ids, partition_ids = [np.array(x) for x in list(zip(*parsed))]
    partition_ids = partition_ids.astype(int)


    king_idx = np.isin(kingdom_ids, kingdom_id)
    part_idx = np.isin(partition_ids, partition_id)
    type_idx = np.isin(type_ids, type_id)

    select_idx = king_idx & part_idx & type_idx
    assert select_idx.sum() >0, 'This id combination does not yield any sequences!'

    #index in numpy, and then return again as lists.
    identifiers_out, sequences_out, labels_out = [list(np.array(x)[select_idx]) for x in [identifiers, sequences, labels]]

    return identifiers_out, sequences_out, labels_out

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

class AbstractThreeLineFastaDataset(Dataset):
    '''Abstract Dataset to load a three-line fasta file.
    Need to implement __getitem__ in child classes, to preprocess sequences as needed.'''
    def __init__(self,
                 data_path: Union[str, Path],
                 tokenizer: Union[str, PreTrainedTokenizer] = 'iupac'
                 ):

        super().__init__()
        
        if isinstance(tokenizer, str):
            from tape import TAPETokenizer
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
        raise NotImplementedError

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

# Process this:
# >P10152|EUKARYA|SP|1
# MVMVLSPLLLVFILGLGLTPVAPAQDDYRYIHFLTQHYDAKPKGRNDEYCFNMMKNRRLTRPCKDRNTFI
# SSSSSSSSSSSSSSSSSSSSSSSOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
# To: 23 (CS = label)
class PointerSentinelThreeLineFastaDataset(AbstractThreeLineFastaDataset):
    def __init__(self,
                data_path: Union[str, Path],
                tokenizer: Union[str, PreTrainedTokenizer] = 'iupac'
                ):
        super().__init__(data_path, tokenizer)

    def _process_labels(self, labels: List[int]) -> int:
        '''Takes a SignalP label sequence and creates a CS position label'''
        sp_pos = np.where(np.array(labels) == 0)[0] #0 is token for S
        if len(sp_pos) == 0:
            cs_pos = len(labels) #target is sentinel
        else: 
            cs_pos = sp_pos.max() #target is CS
        
        return cs_pos

    def __getitem__(self, index):
        item = self.sequences[index]
        labels = self.labels[index]
        token_ids = self.tokenizer.tokenize(item)+ [self.tokenizer.stop_token]
        token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)

        label_ids = self.label_tokenizer.sequence_to_token_ids(labels)
        label = self._process_labels(label_ids)
        return np.array(token_ids), np.array(label)

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(zip(*batch))
        data = torch.from_numpy(pad_sequences(input_ids, 0))
        
        targets = torch.tensor(np.stack(labels))

        return data, targets



class ThreeLineFastaDataset(Dataset):
    """Creates a dataset from a SignalP format 3-line .fasta file.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 tokenizer: Union[str, PreTrainedTokenizer] = 'iupac',
                 add_special_tokens = False,
                 ):

        super().__init__()
        self.add_special_tokens = add_special_tokens
        
        if isinstance(tokenizer, str):
            from tape import TAPETokenizer
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        self.label_tokenizer = SP_label_tokenizer()

        self.data_file = Path(data_path)
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)
        
        self.identifiers, self.sequences, self.labels = parse_threeline_fasta(self.data_file)
        self.global_labels = [x.split('|')[2] for x in self.identifiers]
        

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index):
        item = self.sequences[index]
        labels = self.labels[index]
        global_label = self.global_labels[index]
        sample_weight = self.sample_weights[index] if hasattr(self, 'sample_weights') else None
        kingdom_id =  SIGNALP_KINGDOM_DICT[self.kingdom_ids[index]] if hasattr(self, 'kingdom_ids') else None
        

        if self.add_special_tokens == True:
            token_ids = self.tokenizer.encode(item)
        else: 
            token_ids = self.tokenizer.tokenize(item)# + [self.tokenizer.stop_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)

        label_ids = self.label_tokenizer.sequence_to_token_ids(labels)
        global_label_id = SIGNALP_GLOBAL_LABEL_DICT[global_label]

        input_mask = np.ones_like(token_ids)

        return_tuple = (np.array(token_ids), np.array(label_ids), np.array(input_mask), global_label_id)

        if sample_weight is not None:
            return_tuple = return_tuple + (sample_weight,)
        if kingdom_id is not None:
            return_tuple = return_tuple + (kingdom_id,)

        return return_tuple

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        #unpack the list of tuples
        if  hasattr(self, 'sample_weights') and hasattr(self, 'kingdom_ids'):
            input_ids, label_ids,mask, global_label_ids, sample_weights, kingdom_ids = tuple(zip(*batch))
        elif hasattr(self, 'sample_weights'):
            input_ids, label_ids,mask, global_label_ids, sample_weights = tuple(zip(*batch))
        elif hasattr(self, 'kingdom_ids'):
            input_ids, label_ids,mask, global_label_ids, kingdom_ids = tuple(zip(*batch))
        else:
            input_ids, label_ids,mask, global_label_ids = tuple(zip(*batch))

        data = torch.from_numpy(pad_sequences(input_ids, 0))
        # ignore_index is -1
        targets = torch.from_numpy(pad_sequences(label_ids, -1))
        mask = torch.from_numpy(pad_sequences(mask, 0))
        global_targets = torch.tensor(global_label_ids)
    
        return_tuple = (data, targets, mask, global_targets)
        if  hasattr(self, 'sample_weights'):
            sample_weights = torch.tensor(sample_weights)
            return_tuple = return_tuple + (sample_weights,)
        if hasattr(self, 'kingdom_ids'):
            kingdom_ids = torch.tensor(kingdom_ids)
            return_tuple = return_tuple + (kingdom_ids, )

        return return_tuple

class PartitionThreeLineFastaDataset(ThreeLineFastaDataset):
    '''Creates a dataset from a SignalP format 3-line .fasta file.
    Supports extracting subsets from the input .fasta file.
    Inputs:
        data_path: path to .fasta file
        tokenizer: TAPETokenizer to convert sequences
        partition_id: integer 0-4, which partition to use
        kingdom_id: ['EUKARYA', 'ARCHAEA', 'NEGATIVE', 'POSITIVE']
        type_id: ['LIPO', 'NO_SP', 'SP', 'TAT']
        add_special_tokens: bool, allow tokenizer to add special tokens
        one_versus_all: bool, use all types (only so that i don't have to change the script, totally useless otherwise)
        positive_samples_weight: give a weight to positive samples #NOTE overrides weight file
    Attributes:
        type_id: type_id argument
        partition_id: partition_id argument
        kingdom_id: kingdom_id argument
        identifiers: fasta headers
        sequences: amino acid sequences
        labels: label sequences
        global_labels: global type label for each sequence
        kingdom_ids: kingdom_id for each sequence
        sample_weights: weight for each sequence, computed either from positive_samples weight or sample_weights_path
        balanced_sampling_weights: weights for balanced kingdom sampling, use with WeightedRandomSampler

    '''
    def __init__(self,
                data_path: Union[str, Path],
                sample_weights_path: Union[str, Path] = None,
                tokenizer: Union[str, PreTrainedTokenizer] = 'iupac',
                partition_id: List[str] = [0,1,2,3,4],
                kingdom_id: List[str] = ['EUKARYA', 'ARCHAEA', 'NEGATIVE', 'POSITIVE'],
                type_id: List[str] = ['LIPO', 'NO_SP', 'SP', 'TAT'],
                add_special_tokens = False,
                one_versus_all = False,
                positive_samples_weight = None,
                return_kingdom_ids = True,
                ):
        super().__init__(data_path, tokenizer, add_special_tokens)
        self.type_id = type_id
        self.partition_id = partition_id
        self.kingdom_id = kingdom_id
        if not one_versus_all:
            self.identifiers, self.sequences, self.labels = subset_dataset(self.identifiers, self.sequences, self.labels, partition_id, kingdom_id, type_id)
        else:
            #retain all types
            self.identifiers, self.sequences, self.labels = subset_dataset(self.identifiers, self.sequences, self.labels, partition_id, kingdom_id, ['LIPO', 'NO_SP', 'SP', 'TAT'])
        
        self.global_labels = [x.split('|')[2] for x in self.identifiers]

        if return_kingdom_ids:
            self.kingdom_ids = [x.split('|')[1] for x in self.identifiers]

            count_dict = defaultdict(lambda: 0)
            for x in self.kingdom_ids:
                count_dict[x] +=1

            self.balanced_sampling_weights = [1.0/count_dict[i] for i in self.kingdom_ids]


        if sample_weights_path is not None:
            sample_weights_df = pd.read_csv(sample_weights_path, index_col = 0)
            subset_ids = [x.split('|')[0].lstrip('>') for x in self.identifiers]
            df_subset = sample_weights_df.loc[subset_ids]
            self.sample_weights = list(df_subset['0'])
        elif positive_samples_weight is not None:
            #make weights from global labels
            self.sample_weights = [positive_samples_weight if label in ['SP', 'LIPO', 'TAT'] else 1 for label in self.global_labels]
        #NOTE this is just to make the training script more adaptable without having to change batch handling everytime. Always make weights, 
        # decide in training script whether or not to use
        else:
            self.sample_weights = [1 for label in self.global_labels]



EXTENDED_VOCAB = ['NO_SP_I', 'NO_SP_M', 'NO_SP_O',
                  'SP_S', 'SP_I', 'SP_M', 'SP_O',
                  'LIPO_S', 'LIPO_I', 'LIPO_M', 'LIPO_O',
                  'TAT_S', 'TAT_I', 'TAT_M', 'TAT_O']

EXTENDED_VOCAB_CS = ['NO_SP_I', 'NO_SP_M', 'NO_SP_O',
                  'SP_S', 'SP_C', 'SP_I', 'SP_M', 'SP_O',
                  'LIPO_S', 'LIPO_C', 'LIPO_I', 'LIPO_M', 'LIPO_O',
                  'TAT_S', 'TAT_C', 'TAT_I', 'TAT_M', 'TAT_O']


class LargeCRFPartitionDataset(PartitionThreeLineFastaDataset):
    '''Same as PartitionThreeLineFastaDataset, but converts sequence labels to be used in the large CRF setup.
    Large CRF: each SP type has own states for all possbilities, no shared non-sp states.
    Label conversion is only done in __getitem__, in order not to interfere with filtering functions.
    '''
    def __init__(self,
                data_path: Union[str, Path],
                sample_weights_path = None,
                tokenizer: Union[str, PreTrainedTokenizer] = 'iupac',
                partition_id: List[str] = [0,1,2,3,4],
                kingdom_id: List[str] = ['EUKARYA', 'ARCHAEA', 'NEGATIVE', 'POSITIVE'],
                type_id: List[str] = ['LIPO', 'NO_SP', 'SP', 'TAT'],
                add_special_tokens = False,
                one_versus_all = False,
                positive_samples_weight = None,
                return_kingdom_ids = False,
                make_cs_state = False
                ):
        super().__init__(data_path, sample_weights_path, tokenizer, partition_id, kingdom_id, type_id, add_special_tokens, one_versus_all, positive_samples_weight, return_kingdom_ids)
        self.label_tokenizer = SP_label_tokenizer(EXTENDED_VOCAB_CS if make_cs_state else EXTENDED_VOCAB)
        self.make_cs_state = make_cs_state


    def __getitem__(self, index):
        item = self.sequences[index]
        labels = self.labels[index]
        global_label = self.global_labels[index]
        weight = self.sample_weights[index] if hasattr(self, 'sample_weights') else None
        kingdom_id =  SIGNALP_KINGDOM_DICT[self.kingdom_ids[index]] if hasattr(self, 'kingdom_ids') else None
        

        if self.add_special_tokens == True:
            token_ids = self.tokenizer.encode(item, kingdom_id = self.kingdom_ids[index])
        else: 
            token_ids = self.tokenizer.tokenize(item)# + [self.tokenizer.stop_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)

        #dependent on the global label, convert and tokenize labels
        labels = labels.replace('L','S').replace('T','S') #all sp-same letter, type info comes from global label in next step

        #If make_cs_state, convert position before the cs from 'S' to 'C'
        if self.make_cs_state:
            last_idx = labels.rfind('S')
            if last_idx != -1:
                l = list(labels)
                l[last_idx] = 'C'
                labels = ''.join(l)
            

        converted_labels = [global_label + '_' + lab for lab in labels]
        label_ids = self.label_tokenizer.convert_tokens_to_ids(converted_labels)
        global_label_id = SIGNALP_GLOBAL_LABEL_DICT[global_label]

        input_mask = np.ones_like(token_ids)

        return_tuple = (np.array(token_ids), np.array(label_ids), np.array(input_mask), global_label_id)

        if weight is not None:
            return_tuple = return_tuple + (weight,)
        if kingdom_id is not None:
            return_tuple = return_tuple + (kingdom_id,)

        return return_tuple


#clean implementation without inheritance from base classes. This will hopefully be the definitive one.
class RegionCRFDataset(PartitionThreeLineFastaDataset):
    '''Converts label sequences to array for multi-state crf'''
    def __init__(self,
            data_path: Union[str, Path],
            sample_weights_path = None,
            tokenizer: Union[str, PreTrainedTokenizer] = 'iupac',
            partition_id: List[str] = [0,1,2,3,4],
            kingdom_id: List[str] = ['EUKARYA', 'ARCHAEA', 'NEGATIVE', 'POSITIVE'],
            type_id: List[str] = ['LIPO', 'NO_SP', 'SP', 'TAT', 'LIPOTAT', 'PILIN'],
            add_special_tokens = False,
            label_vocab = None,
            global_label_dict = None,
            one_versus_all = False,
            positive_samples_weight = None,
            return_kingdom_ids = False,
            make_cs_state = False #legacy to not break code when just plugging in this dataset
            ):
        super().__init__(data_path, sample_weights_path, tokenizer, partition_id, kingdom_id, type_id, add_special_tokens, one_versus_all, positive_samples_weight, return_kingdom_ids)
        self.label_vocab = label_vocab #None is fine, process_SP will use default
        self.global_label_dict = global_label_dict if global_label_dict is not None else SIGNALP6_GLOBAL_LABEL_DICT

    def __getitem__(self, index):
        item = self.sequences[index]
        labels = self.labels[index]
        global_label = self.global_labels[index]
        weight = self.sample_weights[index] if hasattr(self, 'sample_weights') else None
        kingdom_id =  SIGNALP_KINGDOM_DICT[self.kingdom_ids[index]] if hasattr(self, 'kingdom_ids') else None

        

        if self.add_special_tokens == True:
            token_ids = self.tokenizer.encode(item, kingdom_id = self.kingdom_ids[index])
        else: 
            token_ids = self.tokenizer.tokenize(item)# + [self.tokenizer.stop_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)



        label_matrix = process_SP(labels, item, sp_type=global_label, vocab=self.label_vocab)
        global_label_id = self.global_label_dict[global_label]

        input_mask = np.ones_like(token_ids)

        #also need to return original tags or cs
        if global_label == 'NO_SP':
            cs = -1
        elif global_label == 'SP':
            cs = labels.rfind('S') +1 #+1 for compatibility. CS reporting uses 1-based instead of 0-based indexing
        elif global_label == 'LIPO':
            cs = labels.rfind('L') +1
        elif global_label == 'TAT':
            cs = labels.rfind('T') +1
        elif global_label == 'TATLIPO':
            cs = labels.rfind('T') +1
        elif global_label == 'PILIN':
            cs = labels.rfind('P') +1
        else:
            raise NotImplementedError(f'Unknown CS defintion for {global_label}')

        return_tuple = (np.array(token_ids), label_matrix, np.array(input_mask), global_label_id, cs)

        if weight is not None:
            return_tuple = return_tuple + (weight,)
        if kingdom_id is not None:
            return_tuple = return_tuple + (kingdom_id,)


        return return_tuple

    #needs new collate_fn, targets need to be padded and stacked.
    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        #unpack the list of tuples
        if  hasattr(self, 'sample_weights') and hasattr(self, 'kingdom_ids'):
            input_ids, label_ids,mask, global_label_ids, cleavage_sites, sample_weights, kingdom_ids = tuple(zip(*batch))
        elif hasattr(self, 'sample_weights'):
            input_ids, label_ids,mask, global_label_ids, sample_weights, cs = tuple(zip(*batch))
        elif hasattr(self, 'kingdom_ids'):
            input_ids, label_ids,mask, global_label_ids, kingdom_ids, cs = tuple(zip(*batch))
        else:
            input_ids, label_ids,mask, global_label_ids = tuple(zip(*batch))

        data = torch.from_numpy(pad_sequences(input_ids, 0))
        
        # ignore_index is -1
        targets = pad_sequences(label_ids, -1)
        targets = np.stack(targets)
        targets = torch.from_numpy(targets) 
        mask = torch.from_numpy(pad_sequences(mask, 0))
        global_targets = torch.tensor(global_label_ids)
        cleavage_sites = torch.tensor(cleavage_sites)

        return_tuple = (data, targets, mask, global_targets, cleavage_sites)
        if  hasattr(self, 'sample_weights'):
            sample_weights = torch.tensor(sample_weights)
            return_tuple = return_tuple + (sample_weights,)
        if hasattr(self, 'kingdom_ids'):
            kingdom_ids = torch.tensor(kingdom_ids)
            return_tuple = return_tuple + (kingdom_ids, )

        return return_tuple