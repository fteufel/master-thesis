import pandas as pd
import numpy as np
import torch
import argparse
import sys
import os
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet

from ensemble_utils import run_data_ensemble

from train_scripts.utils.signalp_dataset import LargeCRFPartitionDataset, SIGNALP_GLOBAL_LABEL_DICT, EXTENDED_VOCAB
from train_scripts.downstream_tasks.metrics_utils import tagged_seq_to_cs_multiclass
from models.sp_tagging_prottrans import BertSequenceTaggingCRF, ProteinBertTokenizer
from models.multi_crf_bert import BertSequenceTaggingCRF as BertSequenceTaggingMultilabelCRF
backtranslate_tokens = [x.split('_')[-1] for x in EXTENDED_VOCAB]




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get ensemble predictions')
    parser.add_argument('--data_path', type=str, default='/work3/felteu/data/signalp_5_data/full/train_set.fasta')
    parser.add_argument('--base_path', type=str, default='/work3/felteu/tagging_checkpoints/bert_crossval/')
    parser.add_argument('--output_path', type=str,default='/work3/felteu/preds')
    parser.add_argument('--multi_label', action='store_true', help='Use Silas CRF BERT')
    parser.add_argument('--full_output', action='store_true')
    parser.add_argument('--partitions', nargs='+', type=int, default=[0,1,2,3,4], help='partitions to use. Default=use all')
    args = parser.parse_args()


    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    #get tokenizer
    tokenizer = ProteinBertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)

    #load dataset
    ds = LargeCRFPartitionDataset(args.data_path,tokenizer=tokenizer,add_special_tokens=True,return_kingdom_ids=True)
    #predict
    dl = torch.utils.data.DataLoader(ds, collate_fn=ds.collate_fn, batch_size=200)

    model = BertSequenceTaggingCRF if not args.multi_label else BertSequenceTaggingMultilabelCRF

    if args.full_output:
        print('Using test partitions')
        print(args.partitions)
        all_global_targets, all_global_probs, all_targets, all_pos_preds, model_names = run_data_ensemble(model,args.base_path, dl, 
                                                                                                        do_not_average=True,
                                                                                                        partitions=args.partitions)
        #each output variable is a tuple of len n_models

        #build a dataframe
        df_dict = {}
        #model_names = ['T0V1', 'T0V2', 'T0V3', 'T0V4','T1V0', 'T1V2', 'T1V3','T1V4','T2V0','T2V1','T2V3','T2V4','T3V0','T3V1','T3V2','T3V4','T4V0','T4V1','T4V2','T4V3']

        for i, name in enumerate(model_names):
            #df_dict[f'loss model {name}'] = all_losses[i]
            df_dict[f'CS model {name}'] = tagged_seq_to_cs_multiclass(all_pos_preds[i], sp_tokens=[3,7,11])
            df_dict[f'p_NO model {name}'] = all_global_probs[i][:,0]
            df_dict[f'p_SPI model {name}'] = all_global_probs[i][:,1]
            df_dict[f'p_SPII model {name}'] = all_global_probs[i][:,2]
            df_dict[f'p_TAT model {name}'] = all_global_probs[i][:,3]


            tagged_seq = np.apply_along_axis(lambda x: ''.join([backtranslate_tokens[i] for i in x]), 1, all_pos_preds[i])
            df_dict[f'tags model {name}'] = tagged_seq

            df_dict['target'] = all_global_targets[0]
            df_dict['CS target'] = tagged_seq_to_cs_multiclass(all_targets[0], sp_tokens = [3,7,11])
            df_dict['identifiers'] = ds.identifiers


            #import IPython; IPython.embed()
            #''.join([backtranslate_tokens[x] for x in all_pos_preds[0][14000]])

            
            df = pd.DataFrame.from_dict(df_dict)

            df[['ID', 'Kingdom', 'Type', 'Partition']] = df['identifiers'].str.lstrip('>').str.split('|', expand=True)
            df.to_csv(os.path.join(args.output_path,'all_model_outputs.csv'))


    #TODO never used this section again, check if still working or delete if not needed
    else:
        all_global_targets, all_global_probs, all_targets, all_pos_preds = run_data_ensemble(model,args.base_path, dl)


        #process loss - average over sequence length
        #all_losses = all_losses.mean(axis=1)

        #make a dataframe
        ds.identifiers #n_sequences
        all_global_targets #n_sequences
        all_global_probs #n_sequences x 4
        all_pos_preds #n_sequences x 7

        cs = tagged_seq_to_cs_multiclass(all_pos_preds, sp_tokens = [3,7,11])


        df = pd.DataFrame.from_dict({ 
                                    'identifiers':ds.identifiers,
                                    'target':all_global_targets,
                                    'CS': cs,
                                    'p_NO' : all_global_probs[:,0],
                                    'p_SPI': all_global_probs[:,1],
                                    'p_SPII': all_global_probs[:,2],
                                    'p_TAT': all_global_probs[:,3],
                                    
                                    })


        df[['ID', 'Kingdom', 'Type', 'Partition']] = df['identifiers'].str.lstrip('>').str.split('|', expand=True)
        #df = df.set_index('ID')

        #rearrange
        df = df[['Kingdom', 'Type', 'Partition', 'p_NO', 'p_SPI', 'p_SPII', 'p_TAT', 'CS', 'target']]
        df.to_csv(os.path.join(args.output_path,'complete_set_probs.csv'))


##code to run plasmodium - TODO cli interface, padding mask, maybe own script
def run_plasmodium(file_path='sp_prediction_experiments/uniprot_manual_plasmodium_sps.tsv', out_name='plasmodium_crossvalidated_predictions.csv'):
    import pandas as pd
    import numpy as np
    from scipy import stats
    from models.sp_tagging_prottrans import BertSequenceTaggingCRF, ProteinBertTokenizer
    from sp_prediction_experiments.investigate_model.ensemble_utils import run_data_ensemble

    tokenizer = ProteinBertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)

    df = pd.read_csv(file_path, sep='\t')

    # tokenizer.tokenizer.pad_token
    seqs = df['Sequence'].apply(lambda x: x[:70]) #truncate
    tokenized = seqs.apply(lambda x: tokenizer.encode(x))
    tokenized =  tokenized.apply(lambda x: x + [0] * (72-len(x))) #pad
    model_input = np.vstack(tokenized)

    res = run_data_ensemble(BertSequenceTaggingCRF, dataloader='x', base_path='/work3/felteu/tagging_checkpoints/bert_crossval/', data_array=model_input, do_not_average=True)
    probs, paths, model_names = res

    #get cs
    pred_cs = []
    for path in paths:
        cs = tagged_seq_to_cs_multiclass(path, sp_tokens = [3,7,11])
        pred_cs.append(cs)

    #get label and probs
    probs =  np.stack(probs).mean(axis=0)
    pred_label = probs.argmax(axis=1)

    pred_cs = np.array(pred_cs) #shape n_models, batch_size
    #TODO need to fix mode to ignore -1 when global label is SP.
    pred_cs = pred_cs.astype(float)  
    pred_cs[pred_cs == -1] = np.nan
    mode, count = stats.mode(pred_cs,axis=0, nan_policy='omit')
    pred_cs = mode.astype(int)



    df['pred CS'] = pred_cs.squeeze()
    df['p_NO'] = probs[:,0]
    df['p_SPI'] = probs[:,1]
    df['p_SPII'] = probs[:,2]
    df['p_TAT'] = probs[:,3]
    df['p_is_SP'] = probs[:,1:].sum(axis=1)

    df['pred label'] =  df[['p_NO', 'p_is_SP']].idxmax(axis=1).apply(lambda x: {'p_is_SP': 'SP', 'p_NO':'Other'}[x])
    #df[['p_NO', 'p_SPI','p_SPII','p_TAT']].idxmax(axis=1) #TODO make this fucntion multi use, not just eukarya
    
    df.to_csv(out_name)
    #import IPython; IPython.embed()
    
    #all_losses, all_global_targets, all_global_probs, all_targets, all_pos_preds, model_names = res

    return df, res

    #average the probs
    #convert all pos_preds to cs and average - or most frequent?
