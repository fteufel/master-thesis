'''
Hybrid model of Bert-CRF and SignalP5.0.
Instead of fine-tuning Bert, sequences are processed through Bert
and fed to a SignalP5Encoder as input features
'''


import torch
import torch.nn as nn
import sys
sys.path.append('..')
from models.multi_tag_crf import CRF
from typing import Tuple
from transformers import BertModel, BertPreTrainedModel, BertTokenizer, PreTrainedModel, PretrainedConfig
import re

import numpy as np
import torch.nn.functional as F 

class SignalPConfig(PretrainedConfig):

    model_type = "signalp"

    def __init__(self,
        dropout_input: float =0.75,
        n_filters: int = 32,
        filter_size: int =3,
        dropout_conv1: float=0.85,
        n_kingdoms: int =4,
        hidden_size: int = 64,
        num_labels: int =9,
        num_global_labels: int =4,
        num_layers: int =1,
        pad_token_id: int = 0,
        bert_checkpoint = 'Rostlab/prot_bert',
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, num_labels=num_labels, **kwargs)
        self.dropout_input = dropout_input
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.dropout_conv1 = dropout_conv1
        self.n_kingdoms = n_kingdoms
        self.hidden_size = hidden_size
        self.num_global_labels = num_global_labels
        self.num_layers = num_layers
        self.bert_checkpoint = bert_checkpoint



class SignalPEncoder(nn.Module):
    '''Encoder of SignalP 5.0 . Taken from Silas' thesis codebase.
    Changes: 
        - args changed to explicit config passing
        - expects pre-embedded inputs.
    '''
    def __init__(self, input_size, dropout_input, n_filters, filter_size, hidden_size, num_layers, dropout_conv1, n_kingdoms=4):
        super().__init__()

        self.n_kingdoms = n_kingdoms
        self.num_layers = num_layers

        self.ReLU1 = nn.ReLU()
        self.ReLU2 = nn.ReLU()


        self.input_dropout = nn.Dropout2d(p=dropout_input)  # keep_prob=0.75
        input_size = input_size
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=n_filters,
                            kernel_size=filter_size, stride=1, padding=filter_size // 2)  # in:20, out=32
        self.conv1_dropout = nn.Dropout2d(p=dropout_conv1)  # keep_prob=0.85  # could this dropout be added directly to LSTM
        self.l_kingdom = nn.Linear(n_kingdoms, hidden_size)  # 4 -> 64

        self.biLSTM = nn.LSTM(input_size=n_filters, hidden_size=hidden_size, num_layers=num_layers,
                            bias=True, batch_first=True, dropout=0.0, bidirectional=True)
        self.conv2 = nn.Conv1d(in_channels=hidden_size * 2, out_channels=n_filters * 2, kernel_size=5,
                            stride=1, padding=5 // 2)  # (128,64)

    def forward(self, embedded_input, kingdom, seq_lengths):

        
        out = embedded_input # [128,70,320]

        out = out.permute(0, 2, 1).float()  # changing [128,70,320] to [128,320,70]
        out = self.input_dropout(out)  # 2D feature map dropout
        out = self.ReLU1(self.conv1(out))  # [128,20,70] -> [128,32,70]
        out = self.conv1_dropout(out)
        bilstminput = out.permute(0, 2, 1).float()  # changing []

        # biLSTM
        kingdom_hot = F.one_hot(kingdom, self.n_kingdoms)  # (indices, depth)
        l_kingdom = self.l_kingdom(kingdom_hot.float())
        kingdom_cell = torch.stack([l_kingdom for i in range(2 * self.num_layers)])  # [2,128,64]

        packed_x = nn.utils.rnn.pack_padded_sequence(bilstminput, seq_lengths.cpu().int(),
                                                    batch_first=True, enforce_sorted=False)  # Pack the inputs ("Remove" zeropadding)
        bi_out, states = self.biLSTM(packed_x, (kingdom_cell, kingdom_cell))  # out [128,70,128]
        # lstm out: last hidden state of all seq elemts which is 64, then 128 because bi. This case the only hidden state. Since we have 1 pass.
        # states is tuple of hidden states and cell states at last t in seq (2,bt,64)
        # bi_out, states = self.biLSTM(packed_x)  # out [128,70,128]
        # forwards out[:, :, :hidden_size] and backwards out[:, :, hidden_size:]

        bi_out, _ = nn.utils.rnn.pad_packed_sequence(bi_out, batch_first=True)  # (batch_size, seq_len, hidden_size*2), out: padded seq, lens
        h_t, c_t = states  # hidden and cell states at last time step, timestep t.

        # h_t is the last hidden state when passed through the seq. Then 1 for forward and 1 for b.
        # h_t[0,5] == bi_out[5][-1][:64], picking sample 5, last forward
        # h_t[1,5] == bi_out[5][0][64:], picking sample 5, last backward

        
        # last assert condition does not hold when setting enforce_sorted=False
        # performance loss should be marginal. Never used packing before for LM experiments anyway, worked too.
        assert bi_out.size(1) == seq_lengths.max()# == seq_lengths[0], "Pad_packed error in size dim"
        packed_out = bi_out.permute(0, 2, 1).float()  # [128, 70, 128] -> [128, 128, 70]

        conv2_out = self.ReLU2(self.conv2(packed_out)) # [128, 128, 70] -> [128, 64, 70]
        return conv2_out



class SignalP5Model(PreTrainedModel):

    config_class = SignalPConfig
    base_model_prefix = "signalp"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        #TODO this is super dirty. Should rather build whole model on BertPretrainedModel base.
        # But then have to write new config class.
        self.bert = BertModel.from_pretrained(config.bert_checkpoint)

        self.encoder = SignalPEncoder(self.bert.config.hidden_size,
                                      config.dropout_input,
                                      config.n_filters,
                                      config.filter_size,
                                      config.hidden_size,
                                      config.num_layers,
                                      config.dropout_conv1,
                                      config.n_kingdoms
                                      )

        # Encoder hidden size to CRF num_labels
        self.ReLU = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=config.n_filters* 2, 
                               out_channels=config.num_labels, 
                               kernel_size=1, 
                               stride=1,
                               padding=1 // 2) 

        # CRF
        self.CRF = CRF(config.num_labels, batch_first=True)

        # Marginal probs to global label
        self.fc_signalP_type = nn.Linear(config.num_labels, config.num_global_labels)

#def forward(self, input_ids, targets=None, kingdom_ids=None, input_mask=None, global_targets=None, **kwargs):
    def forward(self, input_ids, targets=None, kingdom_ids=None, input_mask=None, global_targets=None, return_emissions=False):

        if input_mask is None:
            input_mask = torch.ones_like(input_ids)

        frozen_bert=True
        if frozen_bert:
            with torch.no_grad():
                outputs = self.bert(input_ids, attention_mask = input_mask)
        else:
            outputs = self.bert(input_ids, attention_mask = input_mask)
                
        input_embeds=outputs[0]
        input_embeds, input_mask = self._trim_transformer_output(input_embeds,input_mask)

        seq_lengths = input_mask.sum(dim=1)
        hidden_states = self.encoder(input_embeds, kingdom_ids, seq_lengths)


        aa_class_logits = self.conv3(hidden_states).permute(0, 2, 1).float()


        # SignalP5.0 uses crossentropy on marginal probs instead of crf nll
        #if targets is not None:
        #    log_likelihood = self.CRF(emissions=aa_class_logits,
        #                            tags=targets,
        #                            mask=input_mask.byte(),
        #                            reduction='mean')
        #    loss_crf = -log_likelihood /self.crf_divide
        #else:
        #    loss_crf = 0

        aa_class_soft = self.CRF.compute_marginal_probabilities(emissions=aa_class_logits, mask=input_mask.byte())

        type_mean = (aa_class_soft * input_mask.float().unsqueeze(-1)).sum(dim=1) / seq_lengths.unsqueeze(-1).float()  # mean over classes, only aa in seq_lens 
        sp_type_logits = self.fc_signalP_type(type_mean)  # type_pred_ll = [128, 4], 9 -> 4
        sp_type_soft = F.softmax(sp_type_logits, dim=-1)  # prob. distr. over SignalP type [128,4], 

        # Up to here everything as original. Now make output match BERT-CRF implementation.

        viterbi_paths = self.CRF.decode(emissions=aa_class_logits, mask=input_mask.byte())
        
        #pad the viterbi paths
        max_pad_len = max([len(x) for x in viterbi_paths])
        pos_preds = [x + [-1]*(max_pad_len-len(x)) for x in viterbi_paths] 
        pos_preds = torch.tensor(pos_preds, device = aa_class_soft.device) #Tensor conversion is just for compatibility with downstream metric functions

        outputs = (sp_type_soft,aa_class_soft, pos_preds)


        # Global label loss
        losses = 0
        if global_targets is not None: 
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction = 'mean')
            global_loss = loss_fct(
                sp_type_logits.view(-1, self.config.num_global_labels), global_targets.view(-1))

            losses = losses+ global_loss

        if targets is not None:
            loss_fct = nn.NLLLoss(ignore_index=-1, reduction = 'mean')
            #log_probs = torch.log(aa_class_soft)
            #loss = loss_fct(
            #    log_probs.reshape(-1, self.config.num_labels), targets.view(-1))

            log_likelihood = self.CRF(emissions=aa_class_logits,
                                    tags=targets,
                                    mask=input_mask.byte(),
                                    reduction='mean')
            loss = -log_likelihood #/self.crf_divide
            
            losses = losses+loss
            
        
        if targets is not None or global_targets is not None:
           
                outputs = (losses,) + outputs

        if return_emissions:
            outputs = outputs + (aa_class_logits, input_mask,) #(batch_size, seq_len, num_labels)
        
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