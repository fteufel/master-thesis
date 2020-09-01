'''
Multiple model architectures to predict SPs on top of a LM.
'''

import torch
import torch.nn as nn
import sys
sys.path.append('..')
from models.awd_lstm import ProteinAWDLSTMAbstractModel, ProteinAWDLSTMModel
from models.crf_layer import CRF
from typing import Tuple


class ProteinAWDLSTMForSPTagging(ProteinAWDLSTMAbstractModel):
    '''Simple position-wise classification model.
    Takes inputs with batch_first.

    '''
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ProteinAWDLSTMModel(config = config, is_LM = False)
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.classifier_hidden_size), 
                                                  nn.ReLU(),
                                                  nn.Linear(config.classifier_hidden_size, config.num_labels),
                                                  #nn.Softmax() #TODO crossentropyloss already has this?
                                                )

        #To make a single global prediction of whether a signal peptide is present or not in a protein, 
        #we take the average of the marginal probabilities across the sequence 
        #(nine classes: Sec/SPI signal, Tat/SPI signal, Sec/SPII signal, outer region, inner region, TM in-out, TM out-in, 
        #Sec SPI/Tat SPI cleavage site and Sec/SPII cleavage site) and perform an affine linear transformation into 
        #four classes (Sec/SPI, Sec/SPII, Tat/SPI, Other),ls=Ws[1T∑Tt=1p(yt|x)], 
        #so as to get the logit of a categorical distribution over the presence or not of a signal peptide.
        self.global_classifier = nn.Sequential(nn.Linear(config.num_labels, 4), nn.Softmax(dim = -1))

        self.init_weights()



    def forward(self, input_ids, input_mask=None, targets =None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Predict sequence features.
        Inputs:  input_ids (batch_size, seq_len)
                 targets (batch_size, seq_len). number of distinct values needs to match config.num_labels
        Outputs: (loss: torch.tensor)
                 prediction_scores: raw model outputs (batch_size, seq_len, num_labels)

        '''
        #transpose mask and ids - ProteinAWDLSTMModel is still seq_len_first.
        input_ids = input_ids.transpose(0,1)
        if input_mask is not None:
            input_mask = input_mask.transpose(0,1)
        outputs = self.encoder(input_ids, input_mask)
        sequence_output, _ = outputs
        sequence_output = sequence_output.transpose(0,1) #reshape to batch_first

        prediction_scores = self.classifier(sequence_output)
        pooled_outputs = torch.nn.functional.softmax(prediction_scores, dim = -1).mean(axis =1)
        global_scores = self.global_classifier(pooled_outputs)

        outputs = (global_scores, prediction_scores)


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
            
        return outputs         # = (loss), global_scores, prediction_scores

#25% test
#sentinel model
#multiple training partitions

class RecurrentPointerSentinelmodule(nn.Module):
    '''Wrapper for LSTM with linear layer, because LSTM cannot be used in nn.Sequential.'''
    def __init__(self, input_size, hidden_size, batch_first = False, bidirectional = True, num_layers =1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first =batch_first, bidirectional = bidirectional, num_layers = num_layers)
        self.linear = nn.Linear(2*hidden_size if bidirectional == True else hidden_size, 1)
    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        output = self.linear(lstm_out)
        return output

class ProteinAWDLSTMPointerSentinelModel(ProteinAWDLSTMAbstractModel):
    '''Pointer-Sentinel model to detect cleavage site of the signal peptide.
    Softmax over the full sequence length, with a sentinel position at the end to maximize if no SP is present.
    Sentinel position can be used as binary SP presence label.
    THE EOS token is used as the sentinel. Ensure that tokenizer adds it.
    '''
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ProteinAWDLSTMModel(config = config, is_LM = False)
        #self.sequence_logits = nn.Sequential(nn.Linear(config.hidden_size, config.classifier_hidden_size), 
        #                                          nn.ReLU(),
        #                                          nn.Linear(config.classifier_hidden_size, 1),
                                                  #nn.Softmax() #TODO crossentropyloss already has this?
        #                                        )
        self.sequence_logits = RecurrentPointerSentinelmodule(config.hidden_size, config.classifier_hidden_size, batch_first = True, bidirectional = True, num_layers= 1)
                                                
        self.init_weights()



    def forward(self, input_ids, input_mask=None, targets =None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Predict sequence features.
        Inputs:  input_ids (batch_size, seq_len)
                 targets (batch_size, seq_len). number of distinct values needs to match config.num_labels
        Outputs: (loss: torch.tensor)
                 prediction_scores: raw model outputs (batch_size, seq_len, num_labels)

        '''
        #transpose mask and ids - ProteinAWDLSTMModel is still seq_len_first.
        input_ids = input_ids.transpose(0,1)
        if input_mask is not None:
            input_mask = input_mask.transpose(0,1)
        outputs = self.encoder(input_ids, input_mask)
        sequence_output, _ = outputs
        sequence_output = sequence_output.transpose(0,1) #reshape to batch_first

        prediction_scores = self.sequence_logits(sequence_output) #batch_size, seq_len, 1
        prediction_scores = prediction_scores.squeeze() #get rid of dummy dim - seq_len is my 'feature dimension' - label is position

        outputs = (torch.nn.functional.softmax(prediction_scores, dim = -1), )


        if targets is not None:
            #compute cleavage site detection loss - CrossEntropyLoss includes Softmax.
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.shape[-1]), targets.view(-1))
            
    
            outputs = (loss,) + outputs
            
        return outputs         # = (loss), prediction_probs


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

class ProteinAWDLSTMSequenceTaggingCRF(ProteinAWDLSTMAbstractModel):
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

        self.encoder = ProteinAWDLSTMModel(config = config, is_LM = False)
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
        #transpose mask and ids - ProteinAWDLSTMModel is still seq_len_first.
        input_ids = input_ids.transpose(0,1)
        if input_mask is not None:
            input_mask = input_mask.transpose(0,1)
        outputs = self.encoder(input_ids, input_mask)
        sequence_output, _ = outputs
        sequence_output = sequence_output.transpose(0,1) #reshape to batch_first

        prediction_logits = self.outputs_to_emissions(sequence_output)

        if self.use_crf == True:
            probs, viterbi_paths = self.crf(prediction_logits) #NOTE do not use loss implemented in this layer, so that I can compare directly to use_crf==False
            log_probs = torch.log(probs)
            #pad the viterbi paths
            max_pad_len = max([len(x) for x in viterbi_paths])
            pos_preds = [x + [-1]*(max_pad_len-len(x)) for x in viterbi_paths] 
            #pos_preds = torch.tensor(viterbi_paths,device = probs.device) #NOTE there is no need for this to be on GPU, but amp throws warnings otherwise

            pos_preds = torch.tensor(pos_preds, device = probs.device) #NOTE as tensor just for compatibility with the else case, so always same type
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