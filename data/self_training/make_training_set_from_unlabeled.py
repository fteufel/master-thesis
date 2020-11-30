'''
Make a three-line fasta training set from 
unlabeled Uniprot data (.csv format with PYLUM)

'''
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/")
from models.multi_crf_bert import BertSequenceTaggingCRF, ProteinBertTokenizer
from train_scripts.utils.signalp_dataset import PredictionDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





LABEL_STR_VOCAB = {
                   0: 'I', #'NO_SP_I' : 0,
                   1: 'M', #'NO_SP_M' : 1,
                   2:'O', #'NO_SP_O' : 2,

                   3:'S', #'SP_N' :    3,
                   4:'S', #'SP_H' :    4,
                   5:'S', #'SP_C' :    5,
                   6:'I', #'SP_I' :    6,
                   7:'M', #'SP_M' :    7,
                   8:'O', #'SP_O' :    8,

                    9:'L',#'LIPO_N':   9,
                    10:'L',#'LIPO_H':  10,
                    11:'L',#'LIPO_CS': 11, #conserved 2 positions before the CS are not hydrophobic,but are also not considered a c region
                    12:'O',#'LIPO_C1': 12, #the C in +1 of the CS
                    13:'I',#'LIPO_I':  13,
                    14:'M',#'LIPO_M':  14,
                    15:'O',#'LIPO_O':  15,

                    16:'T',#'TAT_N' :  16,
                    17:'T',#'TAT_RR':  17, #conserved RR marks the border between n,h
                    18:'T',#'TAT_H' :  18,
                    19:'T',#'TAT_C' :  19,
                    20:'I',#'TAT_I' :  20,
                    21:'M',#'TAT_M' :  21,
                    22:'O',#'TAT_O' :  22,

                    23:'T',#'TATLIPO_N' : 23,
                    24:'T',#'TATLIPO_RR': 24,
                    25:'T',#'TATLIPO_H' : 25,
                    26:'T',#'TATLIPO_CS' : 26,
                    27:'O',#'TATLIPO_C1': 27, #the C in +1 of the CS
                    28:'I',#'TATLIPO_I' : 28,
                    29:'M',#'TATLIPO_M' : 29,
                    30:'O',#'TATLIPO_O' : 30,

                    31:'P',#'PILIN_P': 31,
                    32:'O',#'PILIN_CS':32,
                    33:'M',#'PILIN_H': 33,
                    34:'I',#'PILIN_I': 34,
                    35:'M',#'PILIN_M': 35,
                    36:'O',#'PILIN_O': 36,

                    -1:''#pad token
                    }

def convert_viterbi_path_to_label_string(viterbi_path: np.ndarray):
    symbols = [LABEL_STR_VOCAB[i] for i in viterbi_path]
    return "".join(symbols)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--unlabeled_data', type=str)
    parser.add_argument('--model_checkpoint', type=str)
    parser.add_argument('--output_file_name', type=str, default='pred_samples.fasta')
    parser.add_argument('--partition_id', type=int, default=0, help='Partition ID to write to fasta output headers')
    parser.add_argument('--threshold', type=float, default = 0.9,help='Minimum prob of model to accept as training sample')

    args=parser.parse_args()

    ## Load data
    df = pd.read_csv(args.unlabeled_data, sep='\t')
    tokenizer = ProteinBertTokenizer.from_pretrained('/zhome/1d/8/153438/experiments/master-thesis/resources/vocab_with_kingdom', do_lower_case=False)

    ds = PredictionDataset(df, tokenizer)
    dl = torch.utils.data.DataLoader(ds, collate_fn=ds.collate_fn,batch_size=200)

    ## Predict
    model = BertSequenceTaggingCRF.from_pretrained(args.model_checkpoint)
    model.to(device)

    global_prob_list = []
    pos_pred_list = []
    kingdom_list = []
    sequence_list = []
    entry_list = []
    with torch.no_grad():
        for idx, b in tqdm(enumerate(dl)):
            data,mask,kingdoms, sequences, entries = b
            data, mask = data.to(device), mask.to(device)

            global_probs, pos_probs, pos_preds = model(data, input_mask=mask)
            global_prob_list.append(global_probs.detach().cpu().numpy())
            pos_pred_list.append(pos_preds.detach().cpu().numpy())
            kingdom_list.append(kingdoms)
            sequence_list.append(sequences)
            entry_list.append(entries)

            #if idx>2:
            #    break
    

    #process predictions and filter according to threshold
    global_probs =  np.concatenate(global_prob_list)
    pos_preds = np.concatenate(pos_pred_list)
    kingdoms = np.array([k for b in kingdom_list for k in b])
    sequences = np.array([k for b in sequence_list for k in b])
    entries = np.array([k for b in entry_list for k in b])

    above_threshold = global_probs.max(axis=1) >= args.threshold

    global_preds = global_probs[above_threshold].argmax(axis=1)
    pos_preds = pos_preds[above_threshold]
    kingdoms = kingdoms[above_threshold]
    sequences = sequences[above_threshold]
    entries = entries[above_threshold]



    ## Make labels from predictions (from viterbi paths)
    SIGNALP6_GLOBAL_LABEL_DICT = {0:'NO_SP', 1:'SP',2:'LIPO', 3:'TAT', 4:'TATLIPO', 5:'PILIN'}
    with open(args.output_file_name, 'w') as f:

        for idx, entry in enumerate(entries):
            sp_class = SIGNALP6_GLOBAL_LABEL_DICT[global_preds[idx]]
            label = convert_viterbi_path_to_label_string(pos_preds[idx])

            f.write(entry + '|' + kingdoms[idx] + '|' + sp_class + '|'+ str(args.partition_id) + '\n')
            f.write(sequences[idx] + '\n')
            f.write(label +'\n')

    print(f'Finished processing. Created {len(kingdoms)} predicted samples to train on:')
    print(pd.crosstab(kingdoms,global_preds))


if __name__=='__main__':
    main()