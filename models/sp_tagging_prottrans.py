'''
CRF head model for pretrain pfam bert.
'''

import torch
import torch.nn as nn
import sys
sys.path.append('..')
from models.crf_layer import CRF
from typing import Tuple
from transformers import XLNetModel, XLNetPreTrainedModel, XLNetTokenizer, BertModel, BertPreTrainedModel, BertTokenizer
import re


class ProteinXLNetTokenizer():
    '''Wrapper class to take care of different raw sequence format in ProtTrans compared to TAPE.
    ProtTrans expects spaces between all AAs'''
    def __init__(self, *args, **kwargs):
        self.tokenizer = XLNetTokenizer.from_pretrained(*args, **kwargs)

    def encode(self, sequence):
        # Preprocess sequence to ProtTrans format
        sequence = ' '.join(sequence)
        prepro = re.sub(r"[UZOB]", "X", sequence)
        return self.tokenizer.encode(prepro)

    @classmethod
    def from_pretrained(cls, checkpoint , **kwargs):
        return cls(checkpoint, **kwargs)


class RecurrentOutputsToEmissions(nn.Module):
    '''Wrapper for LSTM with linear layer, because LSTM cannot be used in nn.Sequential.'''
    def __init__(self, input_size, hidden_size, num_labels, batch_first = False, bidirectional = True, num_layers =1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first =batch_first, bidirectional = bidirectional, num_layers = num_layers)
        self.linear = nn.Linear(2*hidden_size if bidirectional == True else hidden_size, num_labels)
    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        output = self.linear(lstm_out)
        return output


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


class XLNetSequenceTaggingCRF(XLNetPreTrainedModel):
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
        self.use_rnn = config.use_rnn if hasattr(config, 'use_rnn') else False
        self.lm_output_dropout = nn.Dropout(config.lm_output_dropout if hasattr(config, 'lm_output_dropout') else 0) #for backwards compatbility
        self.lm_output_position_dropout = SequenceDropout(config.lm_output_position_dropout if hasattr(config, 'lm_output_position_dropout') else 0)
        self.global_label_loss_multiplier = config.global_label_loss_multiplier if hasattr(config, 'global_label_loss_multiplier') else 0

        #more crf states, do not use linear layer to get global probs
        self.use_large_crf = config.use_large_crf if hasattr(config, 'use_large_crf') else False
        self.transformer = XLNetModel(config = config)
        self.outputs_to_emissions = nn.Sequential(nn.Linear(config.hidden_size, config.classifier_hidden_size), 
                                                  nn.ReLU(),
                                                  nn.Linear(config.classifier_hidden_size, config.num_labels),
                                                )
        if self.use_rnn:
            self.outputs_to_emissions = RecurrentOutputsToEmissions(config.hidden_size, config.classifier_hidden_size, config.num_labels, batch_first = True)
        self.crf = CRF(num_tags = config.num_labels, batch_first = True)

        self.global_classifier = nn.Sequential(nn.Linear(config.num_labels, config.num_global_labels), nn.LogSoftmax(dim = -1)) #TODO with my use mode this would be binary crossentropy

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets =None, global_targets = None, return_both_losses = False):
        '''Predict sequence features.
        Inputs:  input_ids (batch_size, seq_len)
                 targets (batch_size, seq_len). number of distinct values needs to match config.num_labels
                 global_targets (batch_size)
                 input_mask (batch_size, seq_len). binary tensor, 0 at padded positions
                 return_both_losses: return per_position_loss and global_loss instead of the sum. Use for debugging/separate optimizing
        Outputs: (loss: torch.tensor)
                 global_probs: global label probs (batch_size, num_labels)
                 probs: model probs (batch_size, seq_len, num_labels)
                 pos_preds: best label sequences (batch_size, seq_len)                 
        '''
        outputs = self.transformer(input_ids, attention_mask = input_mask, mems = None) # Returns tuple. pos 0 is sequence output, rest optional.
        sequence_output = outputs[0]
        # trim CLS and SEP token from sequence
        sequence_output = sequence_output[:,1:-1,:]
        if input_mask  is not None:
            input_mask = input_mask[:,1:-1]
        #apply dropouts
        sequence_output = self.lm_output_dropout(sequence_output)
        prediction_logits = self.outputs_to_emissions(sequence_output)

        if self.use_crf == True:
            probs, viterbi_paths = self.crf(prediction_logits, mask = input_mask) #NOTE do not use loss implemented in this layer, so that I can compare directly to use_crf==False
            log_probs = torch.log(probs)
            #pad the viterbi paths
            max_pad_len = max([len(x) for x in viterbi_paths])
            pos_preds = [x + [-1]*(max_pad_len-len(x)) for x in viterbi_paths] 

            pos_preds = torch.tensor(pos_preds, device = probs.device) #NOTE convert to tensor just for compatibility with the else case, so always returns same type
        else:
            log_probs =  torch.nn.functional.log_softmax(prediction_logits, dim = -1)
            probs =  torch.exp(log_probs)
            pos_preds = torch.argmax(probs, dim =1)

        if not self.use_large_crf:
            pooled_outputs = probs.mean(axis =1) #mean over seq_len
            global_log_probs = self.global_classifier(pooled_outputs)
            global_probs = torch.exp(global_log_probs)

        else:
            global_probs = self.compute_global_labels(probs, input_mask)
            global_log_probs = torch.log(global_probs) #for compatbility, when providing global labels. Don't need global labels for training large crf.

        outputs = (global_probs, probs, pos_preds) #+ outputs

        #get the losses
        losses = 0
        if global_targets is not None: 
            loss_fct = nn.NLLLoss(ignore_index=-1)
            global_loss = loss_fct(
                global_log_probs.view(-1, self.num_global_labels), global_targets.view(-1))
            losses = losses + global_loss * self.global_label_loss_multiplier

        if targets is not None:
            loss_fct = nn.NLLLoss(ignore_index=-1)
            loss = loss_fct(
                log_probs.view(-1, self.config.num_labels), targets.view(-1))
            losses = losses + loss
        
        if targets is not None or global_targets is not None:
            if return_both_losses:
                outputs = (loss, global_loss,) + outputs
            else:
                outputs = (losses,) + outputs
        
        #loss, global_probs, pos_probs, pos_preds
        return outputs

    def compute_global_labels(self, probs, mask):
        '''Compute the global labels as sum over marginal probabilities, normalizing by seuqence length.
        For agrregation, the EXTENDED_VOCAB indices from signalp_dataset.py are hardcoded here.'''
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
        spii =global_probs[:, 7:11].sum(dim =1)
        tat = global_probs[:, 11:].sum(dim =1)

        return torch.stack([no_sp, spi, spii, tat], dim =-1)


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
        self.use_crf = config.use_crf
        self.num_global_labels = config.num_global_labels
        self.num_labels = config.num_labels
        self.use_rnn = config.use_rnn
        self.global_label_loss_multiplier = config.global_label_loss_multiplier

        self.transformer = BertModel(config = config)
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
        outputs = self.transformer(input_ids, attention_mask = input_mask) # Returns tuple. pos 0 is sequence output, rest optional.
        sequence_output = outputs[0]
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