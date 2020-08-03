import torch
import torch.nn as nn
from awd_lstm import ProteinAWDLSTMAbstractModel, ProteinAWDLSTMModel


class ProteinAWDLSTMforSPTagging(ProteinAWDLSTMAbstractModel):
    '''Simple position-wise classification model.
    Takes inputs with batch_first.

    '''
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ProteinAWDLSTMModel(config = config, is_LM = False)
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.classifier_hidden_size), 
                                                  nn.ReLU()
                                                  nn.Linear(config.classifier_hidden_size, config.num_labels),
                                                  nn.Softmax() #TODO crossentropyloss already has this?
                                                )
                                                

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets =None):
        '''Predict sequence features.
        Inputs:  input_ids (batch_size, seq_len)
                 targets (batch_size, seq_len). number of distinct values needs to match config.num_labels
        Outputs: (loss: torch.tensor)
                 prediction_scores: raw model outputs (batch_size, seq_len, num_labels)

        '''
        #transpose mask and ids - ProteinAWDLSTMModel is still seq_len_first.
        outputs = self.encoder(input_ids.transpose(0,1), input_mask.transpose(0,1))
        sequence_output, _ = outputs
        sequence_output = sequence_output.transpose(0,1) #reshape to batch_first
        outputs = self.classifier(sequence_output)

         if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(
                prediction_scores.view(-1, self.config.num_labels), targets.view(-1))
            
    
            outputs = (loss,) + outputs
            
        # (loss), prediction_scores
        return outputs