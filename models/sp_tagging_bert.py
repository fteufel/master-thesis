'''
CRF head model for pretrain pfam bert.
'''

import torch
import torch.nn as nn
import sys
sys.path.append('..')
from tape import ProteinBertAbstractModel, ProteinBertModel
from models.crf_layer import CRF
from typing import Tuple



class ProteinLMSequenceTaggingCRF(ProteinBertAbstractModel):
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
        self.use_crf = config.use_crf
        self.num_global_labels = config.num_global_labels
        self.num_labels = config.num_labels
        self.use_rnn = config.use_rnn
        self.global_label_loss_multiplier = config.global_label_loss_multiplier

        self.encoder = ProteinBertModel(config = config)
        self.outputs_to_emissions = nn.Sequential(nn.Linear(config.hidden_size, config.classifier_hidden_size), 
                                                  nn.ReLU(),
                                                  nn.Linear(config.classifier_hidden_size, config.num_labels),
                                                )
        if self.use_rnn:
            self.outputs_to_emissions = RecurrentOutputsToEmissions(config.hidden_size, config.classifier_hidden_size, config.num_labels, batch_first = True)
        self.crf = CRF(num_tags = config.num_labels, batch_first = True)

        self.global_classifier = nn.Sequential(nn.Linear(config.num_labels, config.num_global_labels), nn.LogSoftmax(dim = -1)) #TODO with my use mode this would be binary crossentropy

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets =None, global_targets = None):
        '''Predict sequence features.
        Inputs:  input_ids (batch_size, seq_len)
                 targets (batch_size, seq_len). number of distinct values needs to match config.num_labels
        Outputs: (loss: torch.tensor)
                 global_probs: global label probs (batch_size, num_labels)
                 probs: model probs (batch_size, seq_len, num_labels)
                 pos_preds: best label sequences (batch_size, seq_len)                 
        '''

        outputs = self.encoder(input_ids, input_mask) # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output, _ = outputs
        # trim CLS and SEP token from sequence
        sequence_output = sequence_output[:,1:-1,:]

        prediction_logits = self.outputs_to_emissions(sequence_output)

        if self.use_crf == True:
            probs, viterbi_paths = self.crf(prediction_logits) #NOTE do not use loss implemented in this layer, so that I can compare directly to use_crf==False
            log_probs = torch.log(probs)
            #pad the viterbi paths
            max_pad_len = max([len(x) for x in viterbi_paths])
            pos_preds = [x + [-1]*(max_pad_len-len(x)) for x in viterbi_paths] 

            pos_preds = torch.tensor(pos_preds, device = probs.device) #NOTE convert to tensor just for compatibility with the else case, so always returns same type
        else:
            log_probs =  torch.nn.functional.log_softmax(prediction_logits, dim = -1)
            probs =  torch.exp(log_probs)
            pos_preds = torch.argmax(probs, dim =1)

        pooled_outputs = probs.mean(axis =1) #mean over seq_len
        global_log_probs = self.global_classifier(pooled_outputs)
        global_probs = torch.exp(global_log_probs)

        outputs = (global_probs, probs, pos_preds) #+ outputs

        #get the losses
        losses = 0
        if global_targets is not None: 
            loss_fct = nn.NLLLoss(ignore_index=-1)
            loss = loss_fct(
                global_log_probs.view(-1, self.num_global_labels), global_targets.view(-1))
            losses = losses + loss * self.global_label_loss_multiplier

        if targets is not None:
            loss_fct = nn.NLLLoss(ignore_index=-1)
            loss = loss_fct(
                log_probs.view(-1, self.config.num_labels), targets.view(-1))
            losses = losses + loss
        
        if targets is not None or global_targets is not None:
            outputs = (losses,) + outputs
        
        #loss, global_probs, pos_probs, pos_preds
        return outputs