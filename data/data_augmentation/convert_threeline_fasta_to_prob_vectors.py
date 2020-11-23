'''
Use Bert LM to replace each AA in a sequence with the probability 
vector of BERT LM at this position.
Slide MASK tag along input sequence
Get AA distribution at each position.

This takes advantage of the fact that each seq is only 70 in length,
so small enough to turn the resulting 70 slided mask sequences into
a (seq_len, seq_len+special_tokens) minibatch that can be run at once.

I do not save the labels/kingdoms/ids after conversion. Assume I always
have the fasta file, so do everything based on .fasta and use resulting
indices to get sequence arrays from (n_sequences, seq_len, vocab_size)
saved probability array.
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

    return masked_probs #(seq_len, vocab_size)


def main(input_fasta_file, output_file_name):

    headers, sequences, labels =  parse_threeline_fasta(input_fasta_file)
    print('Loaded data.')

    model = BertForMaskedLM.from_pretrained('Rostlab/prot_bert')
    model.to(device)
    print('Loaded model.')
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert')


    prob_list = []
    for seq in tqdm(sequences):
        lm_probs =  slide_mask(seq, model, tokenizer)
        prob_list.append(lm_probs)

    #import ipdb; ipdb.set_trace()
    prob_array = np.concatenate(prob_list) #n_sequences, seq_len, vocab_size
    ids = [x.lstrip('>').split('|')[0] for x in headers]
    
    #save
    savedict = dict(zip(ids, prob_list))
    np.savez(output_file_name, **savedict)

if __name__ == '__main__':
    input_fasta_file = '../signal_peptides/signalp_updated_data/signalp_6_train_set.fasta'
    output_file_name = 'bert_prob_vectors_signalp_6_train_set.npz'
    main(input_fasta_file, output_file_name)
