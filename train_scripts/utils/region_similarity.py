'''
Compute the cosine similarity between n,h,c regions.
This is used for evaluation as well as potentially as a regularization term.

For direct compatibility with the model, we operate on the sequence tensors after 
tokenization / before backtranslation.

Module is implemented in torch so that I can experiment with adding this to the loss 
/ scaling the loss by this.
Open question, as viterbi decoding is not differentiable.

'''
from typing import Tuple, Union, List
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


AA_SPECIAL_TOKENS = [0, 1, 2, 3, 4, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41] #these are ignored in distance
TAG_PAD_IDX = -1 
MAX_LEN_AA_TOKENS = 30 #regular AAs + special AAs
 # to remove irrelevant indices after bincount
AA_TOKEN_START_IDX = 5 #first aa token
AA_TOKEN_END_IDX = 25  #last aa token

#SIGNALP6_GLOBAL_LABEL_DICT = {'NO_SP':0, 'SP':1,'LIPO':2, 'TAT':3, 'TATLIPO':4, 'PILIN':5}
N_H_IGNORE_CLASSES = [3,4,5]
H_C_IGNORE_CLASSES = [2,4,5] 

DEFAULT_N_TAGS = [3, 9, 16, 23]
DEFAULT_H_TAGS = [4, 10, 18, 25, 33]
DEFAULT_C_TAGS = [5, 19]

def torch_isin(element: torch.LongTensor, test_elements: Union[torch.LongTensor, List[int]]) -> torch.BoolTensor:
    '''torch equivalent of np.isin()
    element is True when value at this position is in test_elements'''

    if type(test_elements) == list:
        test_elements = torch.tensor(test_elements).to(device)

    bool_tensor = (element.unsqueeze(-1) == test_elements).any(-1)

    return bool_tensor



def compute_region_cosine_similarity(tag_sequences: Union[List[torch.LongTensor], torch.LongTensor], 
                                     aa_sequences:Union[List[torch.LongTensor], torch.LongTensor],
                                     n_region_tags: List[int] = None,
                                     h_region_tags: List[int] = None,
                                     c_region_tags: List[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Calculate the cosine similiarities of sequence regions as defined by the tags.
    This function is not aware of ground truth sequence type/global label,
    which means that also e.g. LIPO SPs will have their h-c similarity calculated.
    These need to be replaced with 0/Nan afterwards as needed.

    Inputs:
        tag_sequences: (n_samples, seq_len) tensor 
                        or list of (seq_len,) tensors with AA token indices
        aa_sequences: tokenize: (n_samples, seq_len) tensor 
                        or list of (seq_len,) tensors with region label indices
        n_region_tags: list of label indices that are considered n region
        h_region_tags: list of label indices that are considered h region
        c_region_tags: list of label indices that are considered c region

    Returns:
        (n_samples,) tensor of n-h similarities
        (n_samples,) tensor of h-c similarities
    
    '''
    if n_region_tags is None:
        n_region_tags = DEFAULT_N_TAGS
    if h_region_tags is None:
        h_region_tags = DEFAULT_H_TAGS
    if c_region_tags is None:
        c_region_tags = DEFAULT_C_TAGS

    # check shape of arrays
    assert len(tag_sequences) == len(aa_sequences), 'Need same number of tag and amino acid sequences'

    # TODO check for invalid labels in sequence
    n_h_similarities = []
    h_c_similarities = []

    for i in range(len(tag_sequences)):

        tags = tag_sequences[i]
        aas = aa_sequences[i]

        # Remove special tokens
        aas = aas[~torch_isin(aas, AA_SPECIAL_TOKENS)]
        tags = tags[tags!=TAG_PAD_IDX]

        # Preprocess tags and sequences
        tags = tags[1:] #skip first M
        aas = aas[1:]   #skip first M

        assert len(tags) == len(aas)

        # Get n region
        n_idx = torch_isin(tags, n_region_tags)

        n_aas = aas[n_idx]
        n_aa_freq = torch.bincount(n_aas, minlength = MAX_LEN_AA_TOKENS)  /len(n_aas)
        n_aa_freq = n_aa_freq[AA_TOKEN_START_IDX:AA_TOKEN_END_IDX+1]

        # Get h region
        h_idx = torch_isin(tags, h_region_tags)

        h_aas = aas[h_idx]
        h_aa_freq = torch.bincount(h_aas, minlength = MAX_LEN_AA_TOKENS).float() /len(h_aas)
        h_aa_freq = h_aa_freq[AA_TOKEN_START_IDX:AA_TOKEN_END_IDX+1]

        # Get c region
        c_idx = torch_isin(tags, c_region_tags)

        c_aas = aas[c_idx]
        c_aa_freq = torch.bincount(c_aas, minlength = MAX_LEN_AA_TOKENS).float()  /len(c_aas)
        c_aa_freq = c_aa_freq[AA_TOKEN_START_IDX:AA_TOKEN_END_IDX+1]

        # Compute n vs h
        n_h_similarity =  torch.nn.functional.cosine_similarity(n_aa_freq,h_aa_freq, 0)
        n_h_similarities.append(n_h_similarity)

        # Compute h vs c
        h_c_similarity =  torch.nn.functional.cosine_similarity(h_aa_freq,c_aa_freq, 0)
        h_c_similarities.append(h_c_similarity)

    return torch.stack(n_h_similarities), torch.stack(h_c_similarities)




def class_aware_cosine_similarities(tag_sequences: Union[List[torch.LongTensor], torch.LongTensor, np.ndarray], 
                                    aa_sequences: Union[List[torch.LongTensor], torch.LongTensor, np.ndarray],
                                    class_labels: Union[torch.LongTensor, np.ndarray],
                                    n_region_tags: List[int] = None,
                                    h_region_tags: List[int] = None,
                                    c_region_tags: List[int] = None,
                                    replace_value: float = 0,
                                    op_mode: str = 'torch') -> Tuple[torch.Tensor, torch.Tensor]:
    '''Wrapper for `compute_region_cosine_similiarity`
    Takes care of post-processing kingdoms'''

    assert not (op_mode == 'torch' and replace_value == np.nan), 'Cannot use nan when working in torch'


    if op_mode == 'torch':
        n_h_masking_indices = torch_isin(class_labels, N_H_IGNORE_CLASSES)
        h_c_masking_indices = torch_isin(class_labels, H_C_IGNORE_CLASSES)
        n_h_similarities, h_c_similarities = compute_region_cosine_similarity(tag_sequences,aa_sequences,n_region_tags,h_region_tags,c_region_tags)

    elif op_mode == 'numpy':
        tag_sequences = torch.tensor(tag_sequences).to(device)
        aa_sequences = torch.tensor(aa_sequences).to(device)

        n_h_masking_indices = np.isin(class_labels, N_H_IGNORE_CLASSES)
        h_c_masking_indices = np.isin(class_labels, H_C_IGNORE_CLASSES)        
        with torch.no_grad():
            n_h_similarities, h_c_similarities = compute_region_cosine_similarity(tag_sequences,aa_sequences,n_region_tags,h_region_tags,c_region_tags)
            n_h_similarities = n_h_similarities.detach().cpu().numpy()
            h_c_similarities = h_c_similarities.detach().cpu().numpy()

    else:
        raise NotImplementedError('Valid op_modes are  `torch` and `numpy`')


    #mask
    n_h_similarities[n_h_masking_indices] = replace_value
    h_c_similarities[h_c_masking_indices] = replace_value

    return n_h_similarities, h_c_similarities
