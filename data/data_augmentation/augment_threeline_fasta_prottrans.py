'''
Use Bert LM probabilities to augment the data.
Slide MASK tag along input sequence
Get AA distribution at each position.

This takes advantage of the fact that each seq is only 70 in length,
so small enough to turn the resulting 70 slided mask sequences into
a (seq_len, seq_len+special_tokens) minibatch that can be run at once.
'''
from transformers import BertForMaskedLM, BertTokenizer
import numpy as np
import torch
from scipy.stats import entropy
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_threeline_fasta(file_name):
    with open(file_name, 'r') as f:
        lines = f.read().splitlines()
        identifiers = lines[::3]
        sequences = lines[1::3]
        labels = lines[2::3]
    #return identifiers[-1300:-10], sequences[-1300:-10], labels[-1300:-10]
    return identifiers, sequences, labels

def write_threeline_fasta(identifiers, sequences, labels, file_name):
    with open(file_name, 'w') as f:
        for i in range(len(identifiers)):
            f.write(identifiers[i]+'\n')
            f.write(sequences[i] + '\n')
            f.write(labels[i] + '\n')


def slide_mask(sequence: str, model: BertForMaskedLM, tokenizer:BertTokenizer):
    '''Mask each position in sequence, turn into batch, predict and gather probs at masked positions'''
    sequence_sep = " ".join(sequence)
    input_ids = tokenizer(sequence_sep)['input_ids'] #np.ndarray(seq_len,)


    repeated = np.vstack([input_ids] *len(sequence)) # seq_len, seq_len+special_tokens
    #starting at 0,1 mask tokens until 69,70
    rows, columns = np.diag_indices(len(sequence))
    columns = columns+1
    repeated[rows,columns] = tokenizer.mask_token_id

    with torch.no_grad():
        out = model(torch.tensor(repeated).to(device))

    probs = torch.nn.functional.softmax(out[0], -1).detach().cpu().numpy() #seq_len, seq_len+special_tokens, vocab_size
    masked_probs = probs[rows,columns]
    perplexities = np.exp(entropy(masked_probs, axis=1))

    return masked_probs, perplexities #(seq_len, vocab_size), seq_len



def augment_sequence(sequence:str, model:BertForMaskedLM, tokenizer:BertTokenizer, n_replace=1):

    rev_vocab = dict(zip(tokenizer.vocab.values(), tokenizer.vocab.keys()))

    probs, perplexities = slide_mask(sequence,model,tokenizer)
    ranked = np.argsort(perplexities)[::-1]
    

    pos_replaced = [] #keep track of how many replacements are actually successful

    for i in range(n_replace):
        pos_to_replace = ranked[i]
    
        aa_probs = probs[pos_to_replace]

        # check if max prob is higher 25% -> conserved position, don't change.
        # if max prob lower, than change to aa with 2nd highest prob.
        if aa_probs.max() < 0.25:
            replace_aa = np.argsort(aa_probs)[-2] #2nd highest

            #don't augment with special tokens 
            #don't augment with same token (not guaranteed that most probable token is the correct one, need to check that.)
            if replace_aa not in [0,1,2,3,4] and sequence[pos_to_replace] != rev_vocab[replace_aa]: 
                sequence = list(sequence)
                sequence[pos_to_replace] = rev_vocab[replace_aa]
                sequence = "".join(sequence)
                pos_replaced.append(pos_to_replace)
    
    return sequence, pos_replaced



def main(input_fasta_file, output_file_name, n_replace=1):

    headers, sequences, labels =  parse_threeline_fasta(input_fasta_file)
    print('Loaded data.')

    model = BertForMaskedLM.from_pretrained('Rostlab/prot_bert')
    model.to(device)
    print('Loaded model.')
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert')

    aug_seqs = []
    replaced_pos = [] #for summary stats
    rep_counter = 0
    for seq in tqdm(sequences):
        seq_augmented, pos_replaced =  augment_sequence(seq, model, tokenizer, n_replace)
        aug_seqs.append(seq_augmented)
        rep_counter += len(pos_replaced)
        replaced_pos.append(pos_replaced)


    print(f'Finished augmentation. Replaced {rep_counter} tokens.')
    #TODO work out how i save the replaced pos histogram

    # subset for seqs that were augmented, drop others.
    idx = [len(x) >0 for x in replaced_pos]

    headers_replaced = np.array(headers)[idx]
    sequences_replaced = np.array(sequences)[idx]
    labels_replaced =  np.array(labels)[idx]
    replaced_pos = np.array(replaced_pos)[idx] #array of lists

    write_threeline_fasta(list(headers_replaced), list(sequences_replaced), list(labels_replaced), output_file_name)





    # Everything below here is just to print a summary statistics table
    import pandas as pd

    kingdoms, types = zip(*[x.split('|')[1:3] for x in headers])
    kingdoms, types =  np.array(kingdoms), np.array(types)

    #compute before-after counts
    count_before_dict = {}
    for kingdom in np.unique(kingdoms):
        kingdom_types = types[kingdoms ==kingdom]
        for typ in np.unique(types):
            count_before_dict[kingdom+ ' ' + typ] = (kingdom_types ==typ).sum()


    kingdoms, types = zip(*[x.split('|')[1:3] for x in headers_replaced])
    kingdoms, types =  np.array(kingdoms), np.array(types)

    count_after_dict = {}
    replaced_pos_mean = {}
    replaced_pos_max = {}
    replaced_pos_min = {}
    replaced_sp_num = {}

    # TODO get sp end from labels_replaced
    sp_ends = []
    labeldict = {'NO_SP':'S','SP':'S', 'LIPO':'L', 'TAT':'T','TATLIPO':'T','PILIN':'P'}
    for label,typ in zip(labels_replaced, types):
        end = label.rfind(labeldict[typ]) if typ != 'NO_SP' else len(label)
        sp_ends.append(end)
    sp_ends = np.array(sp_ends)

    
    # compute replaced_pos < sp_end and sum up for each class
    #replaced_in_sp = replaced_pos <= sp_ends

    for kingdom in np.unique(kingdoms):
        kingdom_types = types[kingdoms ==kingdom]
        kingdom_rep_pos =  replaced_pos[kingdoms ==kingdom]
        kingdom_sp_ends = sp_ends[kingdoms == kingdom]
        for typ in np.unique(types):
            count_after_dict[kingdom+ ' ' + typ] = (kingdom_types ==typ).sum()

            pos = kingdom_rep_pos[kingdom_types ==typ] #array of lists
            ends = kingdom_sp_ends[kingdom_types == typ]
            #flatten
            in_sp = np.array([item<= end for (sublist,end) in zip(pos,ends) for item in sublist]) #boolean array, true for each substituted pos that comes before the sp end

            pos = np.array([item for sublist in pos for item in sublist]) #flattened list of all pos

            replaced_pos_mean[kingdom+ ' ' + typ] = pos.mean() if len(pos)>0 else np.nan
            replaced_pos_max[kingdom+ ' ' + typ] = pos.max() if len(pos)>0 else np.nan
            replaced_pos_min[kingdom+ ' ' + typ] = pos.min() if len(pos)>0 else np.nan
            replaced_sp_num[kingdom+ ' ' + typ] = in_sp.sum()

    print('Augmentation statistics')
    pd.set_option('display.max_rows', None)
    print(pd.DataFrame.from_dict([count_before_dict, 
                                  count_after_dict, 
                                  replaced_pos_mean, 
                                  replaced_pos_min, 
                                  replaced_pos_max,
                                  replaced_sp_num]).T.rename({0:'before', 1:'after', 2:'mean pos', 3: 'min pos', 4:'max pos', 5:'rep within SP'}, axis=1))



if __name__ == '__main__':
    input_fasta_file = '../signal_peptides/signalp_updated_data/signalp_6_train_set.fasta'
    output_file_name = 'augmentation_test_1.fasta'
    n_replace = 1
    main(input_fasta_file, output_file_name,n_replace)
