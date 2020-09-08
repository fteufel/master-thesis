'''
Convert data from my format to the one used in udsmprot
(.npy files)
'''
import numpy as np
import pandas as pd
from tape import TAPETokenizer
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter, defaultdict
# taken from UDSMprot, changed to use TAPETokenizer.
def prepare_dataset(path,df_seq,sequence_len_min_tokens=0, sequence_len_max_tokens=0,pad_idx=0, label_itos_in=[], insert_bos=True, insert_eos=False, sequence_output=False, insert_oov=False,max_entries=0,regression=False,max_vocab=60000,min_freq=2, mask_idx=None, df_seq_redundant=None):
    '''
    Creates set of numerical arrays from a dataframe for further processing.
    Parameters:
    path: output path as string
    df_seq: pandas dataframe with data; columns ID (as index), sequence (and optionally label)
    sequence_len_min_tokens: minimum length of tokenized sequence
    sequence_len_max_tokens: maximum length of tokenized sequence (0 for no length restriction)
    tok_itos_in: optional int to string mapping for tokens (to obtain a consistent tokenization with previous pretraining tasks)
    label_itos_in: optional int to string mapping for labels (in case the mapping was already applied in the dataframe) special tokens are _bg_ for background and _none_ for ignoring
    insert_bos: insert bos token (_bos_)
    insert_eos: insert eos token (_eos_)
    pad_idx: ID of the padding token
    insert_oov: insert oov token into vocabulary (otherwise oov will be mapped to padding token)
    
    sequence output: output is a sequence (will add padding token to target labels)
    
    max_entries: return only the first max_entries entries (0 for all)
    regression: labels are continuous for regression task
    max_vocab: only include the most frequent max_vocab tokens in the vocabulary
    min_freq: only include tokens in the vocabulary that have more than min_freq occurrences
    mask_idx: id of mask token (for BERT pretraining) None for none
    
    df_seq_redundant: optional dataframe with redundant sequences
    
    creates in the directory path:
        tok.npy input sequence mapped to integers
        label.npy labels mapped to integers
        tok_itos.npy integer to token mapping
        label_itos.npy integer to label mapping
    '''

    tokenizer = TAPETokenizer(vocab= 'iupac')
    label_none= "_none_" #special label: none (i.e. irrelevant) token for annotation labels e.g. for padding/eos but also for irrelevant phosphorylation site predictions
    label_bg = "_bg_" #special label: background token for annotation labels
    tok_itos_in = list(tokenizer.vocab.keys())

    token_oov="_oov_"
    token_pad="_pad_"
    token_bos= tokenizer.start_token
    token_eos= tokenizer.stop_token
    token_mask=tokenizer.mask_token

    assert(len(np.unique(df_seq.index))==len(df_seq)),"Error in prepare_dataset: df_seq index contains duplicates."
    assert(df_seq_redundant is None or len(np.unique(df_seq_redundant.index))==len(df_seq_redundant)),"Error in prepare_dataset: df_seq_redundant index contains duplicates."
    
    print("\n\nPreparing dataset:", len(df_seq), "rows in the original dataframe.")
    #create target path
    PATH = Path(path)
    PATH.mkdir(exist_ok=True)

    #delete aux. files if they exist
    aux_files = ['val_IDs_CV.npy','val_IDs.npy','train_IDs.npy', 'train_IDs_CV.npy', 'test_IDs.npy','test_IDs_CV.npy']
    for f in aux_files:
        p=PATH/f
        if(p.exists()):
            p.unlink()

    if(df_seq_redundant is None):
        df = df_seq
    else:
        ids_extra = np.setdiff1d(list(df_seq_redundant.index),list(df_seq.index))#in doubt take entries from the original df_seq
        common_columns = list(np.intersect1d(df_seq.columns,df_seq_redundant.columns))
        df = pd.concat([df_seq[common_columns],df_seq_redundant[common_columns].loc[df_seq_redundant.index.intersection(ids_extra)]]) 
        #print("df_seq.columns",df_seq[common_columns].columns,"df_seq_redundant.columns",df_seq_redundant[common_columns].columns,"df.columns",df.columns)
        print("label_itos_in",label_itos_in)

    #label-encode label column
    if "label" in df.columns: #label is specified
        if(regression):
            df["label_enc"]=df.label
        else:#categorical label
            if(len(label_itos_in)>0): #use the given label_itos
                if(isinstance(df.label.iloc[0],list) or isinstance(df.label.iloc[0],np.ndarray)):#label is a list
                    df["label_enc"]=df.label
                else:#label is a single entry
                    df["label_enc"]=df.label.astype('int64')
                label_itos=label_itos_in
            else: #create label_itos
                if(isinstance(df.label.iloc[0],list) or isinstance(df.label.iloc[0],np.ndarray)):#label is a list
                    label_itos = np.sort(np.unique([x for s in list(df.label) for x in s]))
                else:#label is a single entry
                    label_itos=list(np.sort(df.label.unique()).astype(str))
                
                numerical_label=False
                if(not(isinstance(label_itos[0],str))):
                    numerical_label=True
                    label_itos = [str(x) for x in label_itos]
            
            if(sequence_output):#annotation dataset: make sure special tokens are available (and make sure that the label_none item is at pad_idx- otherwise padding won't work as intended)
                label_itos_new = label_itos.copy()
                if(label_itos_new[pad_idx]!=label_none):
                    if(label_none in label_itos_new):
                        label_itos_new.remove(label_none)
                if(not(label_none in label_itos_new)):
                    label_itos_new.insert(pad_idx,label_none)
                if(not(label_bg in label_itos)):
                    label_itos_new.insert(pad_idx+1,label_bg)
                if(len(label_itos_in)>0):#apply new mapping to existing mapped labels
                    label_itoi_transition={idx:label_itos_new.index(label_itos[idx]) for idx in range(len(label_itos))}
                    #label is a list: multilabel classification
                    df["label_enc"]=df.label_enc.apply(lambda x:[label_itoi_transition[y] for y in x])   
                label_itos = label_itos_new
                
            np.save(PATH/"label_itos.npy",label_itos)

            if(len(label_itos_in)==0):#apply mapping to integer
                label_stoi={s:i for i,s in enumerate(label_itos)}
                if(isinstance(df.label.iloc[0],list) or isinstance(df.label.iloc[0],np.ndarray)):#label is a list: multilabel classification
                    df["label_enc"]=df.label.apply(lambda x:[label_stoi[str(y)] for y in x])
                else:#single-label classification
                    df["label_enc"]=df.label.apply(lambda x:label_stoi[str(x)]).astype('int64')

            #one-hot encoding for multilabel classification
            if(sequence_output is False and (isinstance(df.label_enc.iloc[0],list) or isinstance(df.label_enc.iloc[0],np.ndarray))):#multi-label classification
                #one-hot encoding
                def one_hot_encode(x,classes):
                    y= np.zeros(classes)#,dtype=int) #float expected in multilabel_soft_margin_loss
                    for i in x:
                        y[i]=1
                    return y
                df["label_enc"]=df.label_enc.apply(lambda x: one_hot_encode(x,len(label_itos)))

    if(sequence_output):
        label_stoi={s:i for i,s in enumerate(label_itos)}           
            
    #tokenize text (to be parallelized)
    tok = []
    label = []
    ID = []
    for index, row in tqdm(df.iterrows()):
        item_tok = tokenizer.tokenize(row['sequence'])# tokenizer(row['sequence'])
        #                words = tokenizer.tokenize(line.rstrip()) + [tokenizer.stop_token]
        #        for word in words:
        #TODO            tokenlist.append(tokenizer.convert_token_to_id(word))
        if(insert_bos):
            item_tok=[token_bos]+item_tok
        if(insert_eos):
            item_tok=item_tok +[token_eos]
        if(sequence_len_min_tokens>0 and len(item_tok)<sequence_len_min_tokens):
            continue
        if(sequence_len_max_tokens>0 and len(item_tok)>=sequence_len_max_tokens):
            continue
        tok.append(item_tok)
        if("label" in df.columns):
            if(sequence_output is False):
                label.append(row["label_enc"])
            else:
                label_tmp=list(row["label_enc"])
                if(insert_bos):
                    label_tmp = [label_stoi[label_none]] + label_tmp
                if(insert_eos):
                    label_tmp = label_tmp + [label_stoi[label_none]]
                label.append(label_tmp)
        ID.append(index)
        if(max_entries>0 and len(tok)==max_entries):
            break
    
    #NOTE tokenizer has fixed vocab. no need
    #turn into integers
    #if(len(tok_itos_in)==0): #create itos mapping
    #    freq = Counter(p for o in tok for p in o)
    #    tok_itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
    #    if(insert_oov is True):
    #        tok_itos.append(token_oov)
    #    if(mask_idx is None):
    #        tok_itos.insert(pad_idx,token_pad)
    #    else:#order matters
    #        if(pad_idx<mask_idx):
    #            tok_itos.insert(pad_idx,token_pad)
    #            tok_itos.insert(mask_idx,token_mask)
    #        else:
    #            tok_itos.insert(mask_idx,token_mask)
    #            tok_itos.insert(pad_idx,token_pad)
    #else:#use predefined itos mapping
    tok_itos = tok_itos_in
    
    np.save(PATH/"tok_itos.npy",tok_itos)
    print("tok_itos (", len(tok_itos), "items):",list(tok_itos))
    if("label" in df.columns and not(regression)):
        print("label_itos (", len(label_itos), "items):",list(label_itos) if len(label_itos)<20 else list(label_itos)[:20],"" if len(label_itos)<20 else "... (showing first 20 items)")
    
    tok_stoi = defaultdict(lambda:(len(tok_itos) if insert_oov else pad_idx), {v:k for k,v in enumerate(tok_itos)})
    tok_num = np.array([[tok_stoi[o] for o in p] for p in tok])
    np.save(PATH/"tok.npy",tok_num)
    print("Saved",len(tok),"rows (filtered based on sequence length).")
    
    np.save(PATH/"ID.npy",ID)
    if("label" in df.columns):
        if not(regression):
            np.save(PATH/"label.npy",np.array(label).astype(np.int64))
        else:
            np.save(PATH/"label.npy",np.array(label))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert to udsmprot data format')
    parser.add_argument('--data', type = str, default = '/work3/felteu/data//all_organisms_balanced_26082020/eukarya/eukarya_val_full.tsv')
    parser.add_argument('--file_base_name', type = str, default = 'eukarya')
    parser.add_argument('--output_path', type = str, default = '/work3/feluteu/data/all_organisms_balanced_26082020/udsmprot_format/')
    args = parser.parse_args()

    #load, concatenate, tokenize and dump
    df_val = pd.read_csv(os.path.join(args.data, args.file_base_name+'_val_full.tsv'), sep = '\t')
    df_test = pd.read_csv(os.path.join(args.data, args.file_base_name+ '_test_full.tsv'), sep = '\t')
    df_train = pd.read_csv(os.path.join(args.data, args.file_base_name+ '_train_full.tsv'), sep = '\t')

    df_val['sequence'] = df_val['Sequence']
    df_test['sequence'] = df_test['Sequence']
    df_train['sequence'] = df_train['Sequence']

    full_df = pd.concat([df_train, df_test, df_val]).reset_index(drop = True)
    prepare_dataset(args.output_path,full_df,pad_idx=0, insert_bos=False, insert_eos=True, sequence_output=False, insert_oov=False,max_entries=0,regression=False,max_vocab=60000,min_freq=2, mask_idx=None, df_seq_redundant=None)

    #make train, val, test id arrays

    #output: train_IDs.npy/val_IDs.npy/test_IDs.npy (numerical indices designating rows in tok.npy)
    n_train = df_train.shape[0]
    n_test = df_test.shape[0]
    n_val = df_val.shape[0]
    train_ids = np.arange(n_train)
    test_ids = np.arange(n_train, n_test)
    val_ids = np.arange(n_train + n_test, n_val)

    PATH = Path(args.output_path)
    np.save(PATH/"train_IDs.npy",train_ids)
    np.save(PATH/"test_IDs.npy",test_ids)
    np.save(PATH/"val_IDs.npy",val_ids)
