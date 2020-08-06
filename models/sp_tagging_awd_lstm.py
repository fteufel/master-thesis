import torch
import torch.nn as nn
import sys
sys.path.append('..')
from models.awd_lstm import ProteinAWDLSTMAbstractModel, ProteinAWDLSTMModel
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


        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(
                prediction_scores.view(-1, self.config.num_labels), targets.view(-1))
            
    
            outputs = (loss,) + outputs
            
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