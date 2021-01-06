'''
Write the whole cross-validated averaged model as a single module
Probably very big, but let's see if it hits the memory limit.

This transformers branch supports scripting of Bert.
https://github.com/sbrody18/transformers
'''
import sys
import pandas as pd
import numpy as np
sys.path.append('../')
from models.multi_crf_bert import ProteinBertTokenizer
from train_scripts.utils.signalp_dataset import RegionCRFDataset


from typing import List
import torch
from multi_tag_crf_for_viterbi import CRF
from multi_crf_bert_for_export import BertSequenceTaggingCRF


class EnsembleBertCRFModel(torch.nn.Module):
    def __init__(self, bert_checkpoints, crf_checkpoint):
        super().__init__()


        self.berts = torch.nn.ModuleList([BertSequenceTaggingCRF.from_pretrained(ckp) for ckp in bert_checkpoints])

        #self.crf =  CRF()
        # pull CRF config from BertSequenceTaggingCRF config
        self.crf = CRF(num_tags = self.berts[0].config.num_labels, 
                       batch_first=True, 
                       allowed_transitions= self.berts[0].config.allowed_crf_transitions, 
                       allowed_start= self.berts[0].config.allowed_crf_starts, 
                       allowed_end= self.berts[0].config.allowed_crf_ends
                       )
        
        #get CRF weights from berts and average
        start_transitions = [x.crf.start_transitions for x in self.berts]
        transitions = [x.crf.transitions for x in self.berts]
        end_transitions = [x.crf.end_transitions for x in self.berts]

        start_transitions = torch.stack(start_transitions).mean(dim=0)
        transitions = torch.stack(transitions).mean(dim=0)
        end_transitions = torch.stack(end_transitions).mean(dim=0)

        self.crf.start_transitions.data = start_transitions
        self.crf.transitions.data =  transitions
        self.crf.end_transitions.data = end_transitions

        print('Initalized model and averaged weights for viterbi')


    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):

        # get global probs, sequence probs and emissions from berts
        futures = [torch.jit.fork(model, input_ids, input_mask) for model in self.berts]
        results = [torch.jit.wait(fut) for fut in futures]
        print('bert fwd passes done')
        
        #results is list of (global_prob, seq_prob, emissions) tuples
        global_probs, marginal_probs, emissions = zip(*results)

        global_probs_mean = torch.stack(global_probs).mean(dim=0)
        marginal_probs_mean = torch.stack(marginal_probs).mean(dim=0)
        emissions_mean = torch.stack(emissions).mean(dim=0)

        viterbi_paths = self.crf(emissions_mean, input_mask.byte())
        print('viterbi paths done')

        return global_probs_mean, marginal_probs_mean, viterbi_paths



'''
import torch
from complete_model import EnsembleBertCRFModel

model_list = ['test_0_val_1', 'test_0_val_2',  'test_1_val_0', 'test_1_val_2',  'test_2_val_0', 'test_2_val_1']
base_path = '/work3/felteu/tagging_checkpoints/signalp_6/'

dummy_input_ids =  torch.ones(1,73, dtype=int)
dummy_input_mask =  torch.ones(1,73, dtype=int)

model = EnsembleBertCRFModel([base_path+x for x in model_list], None)
model.eval()
model = torch.jit.trace(model, (dummy_input_ids, dummy_input_mask))
model.save('/work3/felteu/ensemble_scripted.pt')
'''


def threeline_fasta_to_df(file_path):
    '''For convenience, convert three-line fasta files to df format,
    so we can reuse all the functions. '''

    ds = RegionCRFDataset(file_path)
    seqs = ds.sequences
    labs = ds.labels
    types = ds.global_labels
    ids = ds.identifiers
    kingdom_id = ds.kingdom_ids

    df =  pd.DataFrame.from_dict({'Sequence':seqs,'label':labs, 'type':types,'id':ids, 'kingdom':kingdom_id})
    return df


def predict_train_set(model):


    tokenizer=ProteinBertTokenizer.from_pretrained("/zhome/1d/8/153438/experiments/master-thesis/resources/vocab_with_kingdom", do_lower_case=False)

    print('Assuming three-line fasta file. 2-line fasta handling not yet implemented')
    df =  threeline_fasta_to_df("/zhome/1d/8/153438/experiments/master-thesis/data/signal_peptides/signalp_updated_data/signalp_6_train_set.fasta")
    seqs = df['Sequence'].apply(lambda x: x[:70]) #truncate
    tokenized = [tokenizer.encode(x, kd) for x,kd in zip(seqs,df.kingdom)]
    tokenized = [x +[0] * (73-len(x)) for x in tokenized] #pad

    input_ids = np.vstack(tokenized)
    input_mask = (input_ids>0) *1

    return input_ids, input_mask

    probs, marginal_probs, viterbi_paths = model(torch.tensor(input_ids[:10]), torch.tensor(input_mask[:10]))

    return probs, marginal_probs, viterbi_paths
    '''
    df['p_NO'] = probs[:,0]
    df['p_SPI'] = probs[:,1]
    df['p_SPII'] = probs[:,2]
    df['p_TAT'] = probs[:,3]
    df['p_TAT'] = probs[:,3]
    df['p_TATLIPO'] = probs[:,4]
    df['p_PILIN'] = probs[:,5]
    df['p_is_SP'] = probs[:,1:].sum(axis=1)

    df['Path'] = viterbi_paths #TODO need to check whether assigning a 2d array like this works
    if args.kingdom=='eukarya':
        df['pred label'] =  df[['p_NO', 'p_is_SP']].idxmax(axis=1).apply(lambda x: {'p_is_SP': 'SP', 'p_NO':'Other'}[x])
    else:
        df['pred label'] =  df[['p_NO', 'p_SPI','p_SPII','p_TAT', 'p_TATLIPO', 'p_PILIN']].idxmax(axis=1).apply(lambda x: {'p_SPI': 'Sec/SPI',
                                                                                                   'p_SPII': 'Sec/SPII', 
                                                                                                   'p_TAT':'Tat/SPI', 
                                                                                                   'p_TATLIPO':'Tat/SPII',
                                                                                                   'p_PILIN':'Sec/SPIII',
                                                                                                   'p_NO':'Other'}[x])
    
    #df = df.drop(['Sequence', 'Signal peptide'], axis=1)
    df.to_csv(args.output_file)
    '''