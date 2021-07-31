'''
SignalP implementation adapted from Silas' thesis.
Implemented as huggingface models so that I can directy replace the Bert-
based models everywhere. Same API, same config names, same forward signatures.

'''
from transformers import PreTrainedModel, PretrainedConfig
import sys
sys.path.append('..')
from models.multi_tag_crf import CRF
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F # PyTorch Utilily functions

#https://www.ncbi.nlm.nih.gov/Class/FieldGuide/BLOSUM62.txt
#                  [ A,  R,  N,  D,  C,  Q,  E,  G,  H,  I,  L,  K,  M,  F,  P,  S,  T,  W,  Y,  V]
BLOSUM = np.array([[ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],
                   [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],
                   [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],
                   [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],
                   [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
                   [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],
                   [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],
                   [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],
                   [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],
                   [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],
                   [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],
                   [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],
                   [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],
                   [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],
                   [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],
                   [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],
                   [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],
                   [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],
                   [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],
                   [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4]])
#http://www.cbs.dtu.dk/biotools/Seq2Logo-2.0/bin/blosum62.txt
#normalized row-wise. embed AAs as rows of this matrix.
#                                    A,      R,      N,      D,      C,      Q,      E,      G,      H,      I,      L,      K,      M,      F,      P,      S,      T,      W,      Y,      V
BLOSUM_normalized = np.array([[ 0.2901, 0.0310, 0.0256, 0.0297, 0.0216, 0.0256, 0.0405, 0.0783, 0.0148, 0.0432, 0.0594, 0.0445, 0.0175, 0.0216, 0.0297, 0.0850, 0.0499, 0.0054, 0.0175, 0.0688],
                              [ 0.0446, 0.3450, 0.0388, 0.0310, 0.0078, 0.0484, 0.0523, 0.0329, 0.0233, 0.0233, 0.0465, 0.1202, 0.0155, 0.0174, 0.0194, 0.0446, 0.0349, 0.0058, 0.0174, 0.0310],
                              [ 0.0427, 0.0449, 0.3169, 0.0831, 0.0090, 0.0337, 0.0494, 0.0652, 0.0315, 0.0225, 0.0315, 0.0539, 0.0112, 0.0180, 0.0202, 0.0697, 0.0494, 0.0045, 0.0157, 0.0270],
                              [ 0.0410, 0.0299, 0.0690, 0.3974, 0.0075, 0.0299, 0.0914, 0.0466, 0.0187, 0.0224, 0.0280, 0.0448, 0.0093, 0.0149, 0.0224, 0.0522, 0.0354, 0.0037, 0.0112, 0.0243],
                              [ 0.0650, 0.0163, 0.0163, 0.0163, 0.4837, 0.0122, 0.0163, 0.0325, 0.0081, 0.0447, 0.0650, 0.0203, 0.0163, 0.0203, 0.0163, 0.0407, 0.0366, 0.0041, 0.0122, 0.0569],
                              [ 0.0559, 0.0735, 0.0441, 0.0471, 0.0088, 0.2147, 0.1029, 0.0412, 0.0294, 0.0265, 0.0471, 0.0912, 0.0206, 0.0147, 0.0235, 0.0559, 0.0412, 0.0059, 0.0206, 0.0353],
                              [ 0.0552, 0.0497, 0.0405, 0.0902, 0.0074, 0.0645, 0.2965, 0.0350, 0.0258, 0.0221, 0.0368, 0.0755, 0.0129, 0.0166, 0.0258, 0.0552, 0.0368, 0.0055, 0.0166, 0.0313],
                              [ 0.0783, 0.0229, 0.0391, 0.0337, 0.0108, 0.0189, 0.0256, 0.5101, 0.0135, 0.0189, 0.0283, 0.0337, 0.0094, 0.0162, 0.0189, 0.0513, 0.0297, 0.0054, 0.0108, 0.0243],
                              [ 0.0420, 0.0458, 0.0534, 0.0382, 0.0076, 0.0382, 0.0534, 0.0382, 0.3550, 0.0229, 0.0382, 0.0458, 0.0153, 0.0305, 0.0191, 0.0420, 0.0267, 0.0076, 0.0573, 0.0229],
                              [ 0.0471, 0.0177, 0.0147, 0.0177, 0.0162, 0.0133, 0.0177, 0.0206, 0.0088, 0.2710, 0.1679, 0.0236, 0.0368, 0.0442, 0.0147, 0.0250, 0.0398, 0.0059, 0.0206, 0.1767],
                              [ 0.0445, 0.0243, 0.0142, 0.0152, 0.0162, 0.0162, 0.0202, 0.0213, 0.0101, 0.1154, 0.3755, 0.0253, 0.0496, 0.0547, 0.0142, 0.0243, 0.0334, 0.0071, 0.0223, 0.0962],
                              [ 0.0570, 0.1071, 0.0415, 0.0415, 0.0086, 0.0535, 0.0708, 0.0432, 0.0207, 0.0276, 0.0432, 0.2781, 0.0155, 0.0155, 0.0276, 0.0535, 0.0397, 0.0052, 0.0173, 0.0328],
                              [ 0.0522, 0.0321, 0.0201, 0.0201, 0.0161, 0.0281, 0.0281, 0.0281, 0.0161, 0.1004, 0.1968, 0.0361, 0.1606, 0.0482, 0.0161, 0.0361, 0.0402, 0.0080, 0.0241, 0.0924],
                              [ 0.0338, 0.0190, 0.0169, 0.0169, 0.0106, 0.0106, 0.0190, 0.0254, 0.0169, 0.0634, 0.1142, 0.0190, 0.0254, 0.3869, 0.0106, 0.0254, 0.0254, 0.0169, 0.0888, 0.0550],
                              [ 0.0568, 0.0258, 0.0233, 0.0310, 0.0103, 0.0207, 0.0362, 0.0362, 0.0129, 0.0258, 0.0362, 0.0413, 0.0103, 0.0129, 0.4935, 0.0439, 0.0362, 0.0026, 0.0129, 0.0310],
                              [ 0.1099, 0.0401, 0.0541, 0.0489, 0.0175, 0.0332, 0.0524, 0.0663, 0.0192, 0.0297, 0.0419, 0.0541, 0.0157, 0.0209, 0.0297, 0.2199, 0.0820, 0.0052, 0.0175, 0.0419],
                              [ 0.0730, 0.0355, 0.0434, 0.0375, 0.0178, 0.0276, 0.0394, 0.0434, 0.0138, 0.0533, 0.0651, 0.0454, 0.0197, 0.0237, 0.0276, 0.0927, 0.2465, 0.0059, 0.0178, 0.0710],
                              [ 0.0303, 0.0227, 0.0152, 0.0152, 0.0076, 0.0152, 0.0227, 0.0303, 0.0152, 0.0303, 0.0530, 0.0227, 0.0152, 0.0606, 0.0076, 0.0227, 0.0227, 0.4924, 0.0682, 0.0303],
                              [ 0.0405, 0.0280, 0.0218, 0.0187, 0.0093, 0.0218, 0.0280, 0.0249, 0.0467, 0.0436, 0.0685, 0.0312, 0.0187, 0.1308, 0.0156, 0.0312, 0.0280, 0.0280, 0.3178, 0.0467],
                              [ 0.0700, 0.0219, 0.0165, 0.0178, 0.0192, 0.0165, 0.0233, 0.0247, 0.0082, 0.1646, 0.1303, 0.0261, 0.0316, 0.0357, 0.0165, 0.0329, 0.0494, 0.0055, 0.0206, 0.2689]])

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



class SignalPEncoder(nn.Module):
    '''Encoder of SignalP 5.0 . Taken from Silas' thesis codebase.
    Changes: 
        - args changed to explicit config passing
        - embedding no longer optional, always expect input_id tokens and
          embed with BLOSUM.    
    '''
    def __init__(self, dropout_input, n_filters, filter_size, hidden_size, num_layers, dropout_conv1, n_kingdoms=4):
        super().__init__()

        self.n_kingdoms = n_kingdoms
        self.num_layers = num_layers

        self.ReLU1 = nn.ReLU()
        self.ReLU2 = nn.ReLU()

        #Add zero vector at pos 0 to embed padding
        embed_weights = np.concatenate([np.zeros((1,20)), BLOSUM_normalized])
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embed_weights), freeze=True)

        self.input_dropout = nn.Dropout2d(p=dropout_input)  # keep_prob=0.75
        input_size = self.embedding.embedding_dim
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=n_filters,
                            kernel_size=filter_size, stride=1, padding=filter_size // 2)  # in:20, out=32
        self.conv1_dropout = nn.Dropout2d(p=dropout_conv1)  # keep_prob=0.85  # could this dropout be added directly to LSTM

        if n_kingdoms >1:
            self.l_kingdom = nn.Linear(n_kingdoms, hidden_size)  # 4 -> 64

        self.biLSTM = nn.LSTM(input_size=n_filters, hidden_size=hidden_size, num_layers=num_layers,
                            bias=True, batch_first=True, dropout=0.0, bidirectional=True)
        self.conv2 = nn.Conv1d(in_channels=hidden_size * 2, out_channels=n_filters * 2, kernel_size=5,
                            stride=1, padding=5 // 2)  # (128,64)

    def forward(self, inp, kingdom, seq_lengths):
        out = inp
        out = self.embedding(out.long()) # [128,70,320]
        out = out.permute(0, 2, 1).float()  # changing [128,70,320] to [128,320,70]
        out = self.input_dropout(out)  # 2D feature map dropout
        out = self.ReLU1(self.conv1(out))  # [128,20,70] -> [128,32,70]
        out = self.conv1_dropout(out)
        bilstminput = out.permute(0, 2, 1).float()  # changing []

        # biLSTM
        if self.n_kingdoms >1:
            kingdom_hot = F.one_hot(kingdom, self.n_kingdoms)  # (indices, depth)
            l_kingdom = self.l_kingdom(kingdom_hot.float())
            kingdom_cell = torch.stack([l_kingdom for i in range(2 * self.num_layers)])  # [2,128,64]
            h_c_tuple = (kingdom_cell, kingdom_cell)
        else:
            h_c_tuple = None

        packed_x = nn.utils.rnn.pack_padded_sequence(bilstminput, seq_lengths.cpu().int(),
                                                    batch_first=True, enforce_sorted=False)  # Pack the inputs ("Remove" zeropadding)
        bi_out, states = self.biLSTM(packed_x, h_c_tuple)  # out [128,70,128]
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



SIGNALP6_CLASS_LABEL_MAP = [[0,1,2], 
                            [3,4,5,6,7,8],  
                            [9, 10 , 11, 12, 13, 14, 15], 
                            [16,17,18,19,20,21,22], 
                            [23,24,25,26,27,28,29,30], 
                            [31,32,33,34,35,36]]

class SignalP5Model(PreTrainedModel):

    config_class = SignalPConfig
    base_model_prefix = "signalp"

    def __init__(self, config):
        super().__init__(config)
        self.config = config


        self.encoder = SignalPEncoder(config.dropout_input,
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
        self.use_signalp6_crf =config.use_signalp6_crf if hasattr(config, 'use_signalp6_crf') else False
        if self.use_signalp6_crf:
            self.class_label_mapping = SIGNALP6_CLASS_LABEL_MAP
        else:
            self.fc_signalP_type = nn.Linear(config.num_labels, config.num_global_labels)

#def forward(self, input_ids, targets=None, kingdom_ids=None, input_mask=None, global_targets=None, **kwargs):
    def forward(self, input_ids, targets=None, targets_bitmap = None, kingdom_ids=None, input_mask=None, global_targets=None, return_emissions=False):

        if input_mask is None:
            input_mask = torch.ones_like(input_ids)

        seq_lengths = input_mask.sum(dim=1)
        hidden_states = self.encoder(input_ids, kingdom_ids, seq_lengths)


        aa_class_logits = self.conv3(hidden_states).permute(0, 2, 1).float()

        aa_class_soft = self.CRF.compute_marginal_probabilities(emissions=aa_class_logits, mask=input_mask.byte())

        type_mean = (aa_class_soft * input_mask.float().unsqueeze(-1)).sum(dim=1) / seq_lengths.unsqueeze(-1).float()  # mean over classes, only aa in seq_lens

        if self.use_signalp6_crf:
            sp_type_soft = self.compute_global_labels_multistate(aa_class_soft,input_mask)
            sp_type_logits = torch.log(sp_type_soft) #computing a loss on this would be uncessesary, just don't provide global labels
        else: 
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

        if targets is not None or targets_bitmap is not None:


            log_likelihood = self.CRF(emissions=aa_class_logits,
                                    tags=targets,
                                    tag_bitmap = targets_bitmap,
                                    mask=input_mask.byte(),
                                    reduction='mean')
            loss = -log_likelihood #/self.crf_divide
            
            losses = losses+loss
            
        
        if targets is not None or global_targets is not None or targets_bitmap is not None:
           
                outputs = (losses,) + outputs

        if return_emissions:
            outputs = outputs + (aa_class_logits, input_mask,) #(batch_size, seq_len, num_labels)
        
        #loss, global_probs, pos_probs, pos_preds
        return outputs




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

        summed_probs = (probs *mask.unsqueeze(-1)).sum(dim =1) #sum probs for each label over axis
        sequence_lengths = mask.sum(dim =1)
        global_probs = summed_probs/sequence_lengths.unsqueeze(-1)

        #aggregate
        no_sp = global_probs[:,0:3].sum(dim=1) 

        spi = global_probs[:,3:7].sum(dim =1)


        if self.num_global_labels >2:
            spii =global_probs[:, 7:11].sum(dim =1)
            tat = global_probs[:, 11:15].sum(dim =1)
            tat_spi = global_probs[:,15:19].sum(dim =1)
            spiii = global_probs[:,19:].sum(dim =1)

            return torch.stack([no_sp, spi, spii, tat, tat_spi, spiii], dim =-1)
        
        else:
            return torch.stack([no_sp, spi], dim =-1)


    def compute_global_labels_multistate(self, probs, mask):
        '''Aggregates probabilities for region-tagging CRF output'''
        if mask is None:
            mask = torch.ones(probs.shape[0], probs.shape[1], device = probs.device)

        summed_probs = (probs *mask.unsqueeze(-1)).sum(dim =1) #sum probs for each label over axis
        sequence_lengths = mask.sum(dim =1)
        global_probs = summed_probs/sequence_lengths.unsqueeze(-1)


        global_probs_list = []
        for class_indices in self.class_label_mapping:
            summed_probs = global_probs[:,class_indices].sum(dim=1)
            global_probs_list.append(summed_probs)

        return torch.stack(global_probs_list, dim =-1)











