import numpy as np

def tagged_seq_to_cs(tagged_seqs: np.ndarray, sp_token = 0):
    '''Convert a sequences of tokens to the index of the cleavage site.
    Inputs:
        tagged_seqs: (batch_size, seq_len) integer array of position-wise labels
        sp_token: label token that indicates a signal peptide
    Returns:
        cs_sites: (batch_size) integer array of last position that is a SP. NaN if no SP present in sequence.
    '''
    def get_last_sp_idx(x: np.ndarray) -> int:
        '''Func1d to get the last index that is tagged as SP. use with np.apply_along_axis. '''
        sp_idx = np.where(x == sp_token)[0]
        max_idx = sp_idx.max() +1 if len(sp_idx)>0 else np.nan #+1, as sequence indexing starts at 1 usually.
        #assert sp_idx.max() - sp_idx.min() == len(sp_idx) #assert continuity. CRF should actually guarantee that (if learned correcty)
        return max_idx

    cs_sites = np.apply_along_axis(get_last_sp_idx, 1, tagged_seqs)
    return cs_sites