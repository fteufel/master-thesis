
import torch
import torch.nn as nn
import sys
sys.path.append('..')
from models.multi_tag_crf import CRF
from typing import Tuple
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
import re

class SequenceDropout(nn.Module):
    '''Layer zeroes full hidden states in a sequence of hidden states'''
    def __init__(self, p = 0.1, batch_first = True):
        super().__init__()
        self.p = p
        self.batch_first = batch_first
    def forward(self, x):
        if not self.training or self.dropout ==0:
            return x

        if not self.batch_first:
            x = x.transpose(0,1)
        #make dropout mask
        mask = torch.ones(x.shape[0], x.shape[1], dtype = x.dtype).bernoulli(1-self.dropout) # batch_size x seq_len
        #expand
        mask_expanded =  mask.unsqueeze(-1).repeat(1,1,16)
        #multiply
        after_dropout = mask_expanded * x
        if not self.batch_first:
            after_dropout = after_dropout.transpose(0,1)

        return after_dropout
        
class ProteinBertTokenizer():
    '''Wrapper class to take care of different raw sequence format in ProtTrans compared to TAPE.
    ProtTrans expects spaces between all AAs'''
    def __init__(self, *args, **kwargs):
        self.tokenizer = BertTokenizer.from_pretrained(*args, **kwargs)

    def encode(self, sequence):
        # Preprocess sequence to ProtTrans format
        sequence = ' '.join(sequence)
        prepro = re.sub(r"[UZOB]", "X", sequence)
        return self.tokenizer.encode(prepro)

    @classmethod
    def from_pretrained(cls, checkpoint , **kwargs):
        return cls(checkpoint, **kwargs)

class BertSequenceTaggingCRF(BertPreTrainedModel):
    '''Sequence tagging and global label prediction model (like SignalP).
    LM output goes through a linear layer with classifier_hidden_size before being projected to num_labels outputs.
    These outputs then either go into the CRF as emissions, or to softmax as direct probabilities.
    config.use_crf controls this.

    Inputs are batch first.
       Loss is sum between global sequence label crossentropy and position wise tags crossentropy.
       Optionally use CRF.

    '''
    def __init__(self, config):
        super().__init__(config)
        self.num_global_labels = config.num_global_labels if hasattr(config, 'num_global_labels') else config.num_labels
        self.num_labels = config.num_labels
        self.lm_output_dropout = nn.Dropout(config.lm_output_dropout if hasattr(config, 'lm_output_dropout') else 0) #for backwards compatbility
        self.lm_output_position_dropout = SequenceDropout(config.lm_output_position_dropout if hasattr(config, 'lm_output_position_dropout') else 0)

        self.use_kingdom_id = config.use_kingdom_id if hasattr(config, 'use_kingdom_id') else False
        self.crf_scaling_factor = config.crf_scaling_factor if hasattr(config, 'crf_scaling_factor') else 1
        self.use_large_crf = config.use_large_crf #TODO legacy for get_metrics, no other use.
        self.bert = BertModel(config = config)

        if self.use_kingdom_id:
            self.kingdom_embedding = nn.Embedding(4, config.kingdom_embed_size)

       
        self.outputs_to_emissions = nn.Linear(config.hidden_size if self.use_kingdom_id is False else config.hidden_size+config.kingdom_embed_size, 
                                                  config.num_labels)
        

        self.crf = CRF(num_tags = config.num_labels, batch_first=True)
        #self.CRF = CRF(self.args.n_classes, batch_first=True, include_start_end_transitions=self.args.crf_priors, constrain_every=self.args.crf_transition_constraint, allowed_transitions=allowed_transitions, allowed_start=allowed_start, allowed_end=allowed_end)

        self.crf_input_length = 70 #TODO make this part of config if needed. Now it's for cases where I don't control that via input data or labels.

        self.init_weights()


    def forward(self, input_ids = None, kingdom_ids = None, input_mask=None, targets =None, global_targets = None, return_both_losses = False, inputs_embeds = None,
                sample_weights = None):
        '''Predict sequence features.
        Inputs:  input_ids (batch_size, seq_len)
                 kingdom_ids (batch_size) :  [0,1,2,3] for eukarya, gram_positive, gram_negative, archaea
                 targets (batch_size, seq_len). number of distinct values needs to match config.num_labels
                 global_targets (batch_size)
                 input_mask (batch_size, seq_len). binary tensor, 0 at padded positions
                 return_both_losses: return per_position_loss and global_loss instead of the sum. Use for debugging/separate optimizing
                 input_embeds: Optional instead of input_ids. Start with embedded sequences instead of token ids.
                 sample_weights (batch_size) float tensor. weight for each sequence to be used in cross-entropy.

        
        Outputs: (loss: torch.tensor)
                 global_probs: global label probs (batch_size, num_labels)
                 probs: model probs (batch_size, seq_len, num_labels)
                 pos_preds: best label sequences (batch_size, seq_len)                 
        '''
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        outputs = self.bert(input_ids, attention_mask = input_mask, inputs_embeds = inputs_embeds) # Returns tuple. pos 0 is sequence output, rest optional.
        
        sequence_output = outputs[0]


        sequence_output, input_mask = self._trim_transformer_output(sequence_output, input_mask) #this takes care of CLS and SEP, pad-aware
        if targets is not None:
            sequence_output = sequence_output[:,:targets.shape[1], :] #this removes extra residues that don't go to CRF
            input_mask =  input_mask[:,:targets.shape[1]]
        else:
            sequence_output = sequence_output[:,:self.crf_input_length, :]
            input_mask = input_mask[:,:self.crf_input_length]
        
        #apply dropouts
        sequence_output = self.lm_output_dropout(sequence_output)

        #add kingdom ids
        if self.use_kingdom_id == True:
            ids_emb = self.kingdom_embedding(kingdom_ids) #batch_size, embed_size
            ids_emb = ids_emb.unsqueeze(1).repeat(1,sequence_output.shape[1],1) #batch_size, seq_len, embed_size
            sequence_output = torch.cat([sequence_output, ids_emb], dim=-1)

        prediction_logits = self.outputs_to_emissions(sequence_output)


        #CRF
        if targets is not None:
            log_likelihood = self.crf(emissions=prediction_logits, tags=targets, tag_bitmap = None, mask = input_mask.byte(), reduction='mean')
            neg_log_likelihood = -log_likelihood *self.crf_scaling_factor
        else:
            neg_log_likelihood = 0
        probs = self.crf.compute_marginal_probabilities(emissions=prediction_logits, mask=input_mask.byte())

        global_probs = self.compute_global_labels(probs, input_mask)
        global_log_probs = torch.log(global_probs) #for compatbility, when providing global labels. Don't need global labels for training large crf.

        # TODO 
        preds = self.predict_global_labels(global_probs, kingdom_ids, weights=None)
        # from preds, make initial sequence label vector
        init_states = self.inital_state_labels_from_global_labels(preds)


        viterbi_paths = self.crf.decode(emissions=prediction_logits, mask=input_mask.byte())
        
        #pad the viterbi paths
        max_pad_len = max([len(x) for x in viterbi_paths])
        pos_preds = [x + [-1]*(max_pad_len-len(x)) for x in viterbi_paths] 
        pos_preds = torch.tensor(pos_preds, device = probs.device) #NOTE convert to tensor just for compatibility with the else case, so always returns same type




        outputs = (global_probs, probs, pos_preds) #+ outputs



        #get the losses
        losses = neg_log_likelihood

        if global_targets is not None: 
            loss_fct = nn.NLLLoss(ignore_index=-1, reduction = 'none' if sample_weights is not None else 'mean')
            global_loss = loss_fct(
                global_log_probs.view(-1, self.num_global_labels), global_targets.view(-1))

            if sample_weights is not None:
                global_loss = global_loss*sample_weights
                global_loss = global_loss.mean()
            
            losses = losses+ global_loss
            
        
        if targets is not None or global_targets is not None:
           
                outputs = (losses,) + outputs
        
        #loss, global_probs, pos_probs, pos_preds
        return outputs

    @staticmethod
    def _trim_transformer_output(hidden_states, input_mask):
        '''Helper function to remove CLS, SEP tokens after passing through transformer'''
        
        #remove CLS
        hidden_states = hidden_states[:,1:,:]
        

        if input_mask is not None:

            input_mask = input_mask[:,1:]
            #remove SEP - hidden states are padded at end!
            true_seq_lens = input_mask.sum(dim =1) -1 #-1 for SEP

            mask_list = []
            output_list = []
            for i in range(input_mask.shape[0]):
                mask_list.append(input_mask[i, :true_seq_lens[i]])
                output_list.append(hidden_states[i,:true_seq_lens[i],:])

            mask_out = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True)
            hidden_out = torch.nn.utils.rnn.pad_sequence(output_list, batch_first=True)
        else:
            hidden_out = hidden_states[:,:-1,:]
            mask_out = None

        return  hidden_out, mask_out
        

    def compute_global_labels(self, probs, mask):
        '''Compute the global labels as sum over marginal probabilities, normalizing by seuqence length.
        For agrregation, the EXTENDED_VOCAB indices from signalp_dataset.py are hardcoded here.
        If num_global_labels is 2, assume we deal with the sp-no sp case.
        TODO refactor, implicit handling of eukarya-only and cs-state cases is hard to keep track of
        '''
        #probs = b_size x seq_len x n_states tensor 
        #Yes, each SP type will now have 4 labels in the CRF. This means that now you only optimize the CRF loss, nothing else. 
        # To get the SP type prediction you have two alternatives. One is to use the Viterbi decoding, 
        # if the last position is predicted as SPI-extracellular, then you know it is SPI protein. 
        # The other option is what you mention, sum the marginal probabilities, divide by the sequence length and then sum 
        # the probability of the labels belonging to each SP type, which will leave you with 4 probabilities.
        if mask is None:
            mask = torch.ones(probs.shape[0], probs.shape[1], device = probs.device)

        #TODO check unsqueeze ops for division/multiplication broadcasting
        summed_probs = (probs *mask.unsqueeze(-1)).sum(dim =1) #sum probs for each label over axis
        sequence_lengths = mask.sum(dim =1)
        global_probs = summed_probs/sequence_lengths.unsqueeze(-1)

        #aggregate
        no_sp = global_probs[:,0:3].sum(dim=1) 

        spi = global_probs[:,3:7].sum(dim =1)


        if self.num_global_labels >2:
            spii =global_probs[:, 7:11].sum(dim =1)
            tat = global_probs[:, 11:].sum(dim =1)

            #When using extra state for CS, different indexing
            if self.num_labels == 18:
                spi = global_probs[:, 3:8].sum(dim =1)
                spii = global_probs[:, 8:13].sum(dim =1)
                tat = global_probs[:,13:].sum(dim =1)


            return torch.stack([no_sp, spi, spii, tat], dim =-1)

        
        else:
            return torch.stack([no_sp, spi], dim =-1)


    @staticmethod
    def predict_global_labels(probs, kingdom_ids, weights= None):
        '''Given probs from compute_global_labels, get prediction.
        Takes care of summing over SPII and TAT for eukarya, and allows reweighting of probabilities.'''

        eukarya_idx = torch.where(kingdom_ids == 0)[0]
        summed_sp_probs = probs[eukarya_idx, 1:].sum(dim=1)
        #update probs for eukarya
        probs[eukarya_idx,1] = summed_sp_probs
        probs[eukarya_idx, 2:] = 0

        #reweight
        if weights is not None:
            probs = probs*weights #TODO check broadcasting, think should just work, 1 axis agrees
        #predict
        preds = probs.argmax(dim=1)

        return preds


    #conversion logic derived from this, hardcoded because never changes.
    #from train_scripts.utils.signalp_dataset import SIGNALP_GLOBAL_LABEL_DICT, EXTENDED_VOCAB
    #EXTENDED_VOCAB = ['NO_SP_I', 'NO_SP_M', 'NO_SP_O',
    #              'SP_S', 'SP_I', 'SP_M', 'SP_O',
    #              'LIPO_S', 'LIPO_I', 'LIPO_M', 'LIPO_O',
    #              'TAT_S', 'TAT_I', 'TAT_M', 'TAT_O']
    #SIGNALP_GLOBAL_LABEL_DICT = {'NO_SP':0, 'SP':1,'LIPO':2, 'TAT':3}
    #SIGNALP_KINGDOM_DICT = {'EUKARYA': 0, 'POSITIVE':1, 'NEGATIVE':2, 'ARCHAEA':3}
    #GLOBAL_STATE_SEQ_STATE_MAP= {0:0, 1:3, 2:7, 3:11}

    @staticmethod
    def inital_state_labels_from_global_labels(preds):

        initial_states = torch.zeros_like(preds)
        #update torch.where((testtensor==1) | (testtensor>0))[0] #this syntax would work.
        initial_states[preds == 0] = 0
        initial_states[preds == 1] = 3
        initial_states[preds == 2] = 7
        initial_states[preds == 3] = 11
        
        return initial_states