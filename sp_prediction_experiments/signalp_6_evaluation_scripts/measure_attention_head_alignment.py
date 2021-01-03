'''
Run the attention alignment metric on all layers+attention heads,
for all labels we have.

Cannot run this with normal python, takes forever.
Alternatives: - multiprocessing (failed a few times, hard to debug)
              - compile the core function with numba.

On HPC: module load python3 numba/0.35.0-python-3.6.2
'''
import h5py
import pickle
import numpy as np
import argparse
from tqdm import tqdm
#rng = np.random.default_rng()

from numba import njit, prange
@njit(parallel=True)
def compute_attention_alignment_individual(attention_head, position_labels, label_to_score, threshold=0.3, shuffle=False):
    '''Implementation of Eq. 1 from the biology bertology paper, for the individual token case.
    attention_head : (batch_size, seq_len, seq_len), one attention head for all samples of dataset
    position_labels: (batch_size, seq_len) position-wise labels for all samples with property info
    label_to_score: the label in position_labels that defines the presence of the property to measure
    shuffle: boolean flag to shuffle the attention weights. Was used as a baseline in the paper
    
    Need to make sure seq_lens of labels and attentions agree (mind special tokens). Easier to fix that
    before than to work with ragged arrays.
    Drop first 2 tokens + last token of attention.
    Pad labels to 70.
    This is a dirty fix, but as padding is no-op it should work fine.
    Alternatively, can run this function with padding as label_to_score to figure out
    whether it is truly no-op.
    '''
    # attention head seq dim needs to match label seq dim
    assert position_labels.shape[1] == attention_head.shape[1]
    assert position_labels.shape[0] == attention_head.shape[0]
    

    #attention_head: (batch_size x seq_len x seq_len)
    total_count = 0
    aligned_count = 0
    #for x in tqdm(range(attention_head.shape[0]), position =3, desc ='samples', leave =False): #for all batches
    for x in prange(attention_head.shape[0]):
        attn_map = attention_head[x].copy()

        #TODO numba doesn't like shuffling
        #if shuffle ==True:
            #shuffle attention from token to other tokens, i->j : shuffle dim 0
        #    np.random.shuffle(attn_map, axis=0)
        for i in range(attention_head.shape[1]):
            for j in range(attention_head.shape[2]):
                
                attn = attn_map[i,j]
                
                if attn > threshold:
                    pos_prop = position_labels[x,j]
                    total_count += 1 #found a high attention weight, add to total count
                    if pos_prop == label_to_score:
                        aligned_count += 1 # found an aligned high attention weight, add to aligned count
                        

    return aligned_count, total_count


def measure_all_alignments(attention_heads, labels):

    # labels are in multi-tag integer id format - 
    # means there are e.g. 4 different ints that all mean n-region.
    # ignore that for now, can aggregate this from final results.
    total_dict = {}
    aligned_dict = {}

    # set up dict of arrays to collect results in
    #two dicts - one for total high attention counts, one for aligned
    #dict has labels as keys, each value is a n_layers x n_heads array
    for label in np.unique(labels):
        aligned = np.zeros((attention_heads.shape[0], attention_heads.shape[2]))
        total = np.zeros((attention_heads.shape[0], attention_heads.shape[2]))
        total_dict[label] = total
        aligned_dict[label] = aligned

        


    # shape (29, 5896, 16, 73, 73) num_layers, num_samples, attn_heads, seq, seq
    for layer_id in tqdm(range(attention_heads.shape[0]), position =0, desc ='layers', leave =True):
        for head_id in tqdm(range(attention_heads.shape[2]), position =1, desc ='heads',leave=False):

            attn_head =  attention_heads[layer_id,:,head_id,2:-1,2:-1] #remove first 2 and last pos #NOTE for non-pretrained bert, change this to first 1

            for label in tqdm(np.unique(labels), position =3, desc ='labels', leave=False):
                
                aligned_count, total_count = compute_attention_alignment_individual(attn_head, labels, label_to_score=label)

                aligned_dict[label][layer_id,head_id] = aligned_count
                total_dict[label][layer_id,head_id] = total_count

    return aligned_dict, total_dict


import multiprocessing
def measure_all_alignments_parallel(attention_heads, labels):
    '''
    Same process as above, but implemented using multiprocessing.
    '''

    #out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))

    #Hence an iterable of [(1,2), (3, 4)] results in [func(1,2), func(3,4)]
    #make iterable
    arglist = []
    print('Preparing function inputs...')
    attention_heads = attention_heads[()]
    for label in tqdm(np.unique(labels), position =0, desc ='labels'):
        for layer_id in tqdm(range(attention_heads.shape[0]), position =1, desc ='layers', leave =False):
            for head_id in tqdm(range(attention_heads.shape[2]), position =2, desc ='heads',leave=False):
                attn_head =  attention_heads[layer_id,:,head_id,2:-1,2:-1]
                arglist.append((attn_head, labels, label)) #attn_head, labels, label_to_score

    #outs = pool.starmap(compute_attention_alignment_individual,arglist)
    print('Spawning processes...')
    with multiprocessing.Pool() as pool:
        # returns list of length arglist, containing tuples
        outs = list(tqdm(pool.starmap(compute_attention_alignment_individual,arglist), total=len(arglist)))
    
    aligned, total = zip(*outs)


    # same iteration again to parse results
    print('parse results into arrays')
    total_dict = {}
    aligned_dict = {}

    idx = 0
    for label in np.unique(labels):

        # set up (layers, heads) arrays to collect results
        aligned = np.zeros((attention_heads.shape[0], attention_heads.shape[2]))
        total = np.zeros((attention_heads.shape[0], attention_heads.shape[2]))

        for layer_id in range(attention_heads.shape[0]):
            for head_id in range(attention_heads.shape[2]):

                aligned[layer_id,head_id] = aligned[idx]
                total[layer_id,head_id] = total[idx]
                idx += 1 

        total_dict[label] = total
        aligned_dict[label] = aligned

    return aligned_dict, total_dict




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention_heads', type=str, default= 'experiments_results/bertology/bert_finetuned_t0v1_attentions.hdf5')
    parser.add_argument('--prediction_outputs', type=str, default = 'experiments_results/bertology/bert_finetuned_t0v1_prediction_outputs.pkl')

    parser.add_argument('--output_file_prefix', type=str, default = 'experiments_results/bertology/t0v1', help='path to dir+prefix of files')
    args = parser.parse_args()


    attention_heads = h5py.File(args.attention_heads, 'r')['attn_heads']

    prediction_outputs = pickle.load(open(args.prediction_outputs, 'rb'))

    aligned_weights, total_weights = measure_all_alignments(attention_heads, prediction_outputs['pos_preds'] )

    with open(args.output_file_prefix + '_attention_alignment_results.pkl', 'wb') as f:
        pickle.dump((aligned_weights, total_weights), f)