#AWD-LSTM implementation based on repo from paper
#Huggingface-like API, based on TAPE
#Felix June 2020
# NOTE LSTM output sizes (as give by system are not true, because the operations are performed at once for all gates, with weigts from the same linear layer)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import typing
from typing import List, Tuple, Dict
import logging
import warnings
import math
from tape.models.modeling_utils import ProteinConfig, ProteinModel
import sys
sys.path.append('..')
from models.modeling_utils import CRFSequenceTaggingHead, SimpleMeanPooling

logger = logging.getLogger(__name__)

#This is just here for compatibility, otherwise useless
URL_PREFIX = "https://s3.amazonaws.com/proteindata/pytorch-models/"
AWDLSTM_PRETRAINED_CONFIG_ARCHIVE_MAP: Dict[str, str] = {}
AWDLSTM_PRETRAINED_MODEL_ARCHIVE_MAP: Dict[str, str] = {}


class ProteinAWDLSTMConfig(ProteinConfig):
    pretrained_config_archive_map = AWDLSTM_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size: int = 30,
                 input_size: int = 400,
                 hidden_size: int = 1150,
                 num_hidden_layers: int = 3,
                 dropout_prob: float = 0.4,
                 hidden_dropout_prob: float = 0.3,
                 embedding_dropout_prob: float = 0.1,
                 input_dropout_prob: float = 0.65,
                 weight_dropout_prob: float = 0.5,
                 beta: float = 1 ,
                 alpha: float = 2,
                 reset_token_id: int = None,
                 bidirectional = False,
                 batch_first: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.input_size = input_size #is the embedding layer size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_prob = dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.embedding_dropout_prob = embedding_dropout_prob
        self.input_dropout_prob = input_dropout_prob
        self.weight_dropout_prob = weight_dropout_prob
        self.alpha = alpha
        self.beta = beta
        self.reset_token_id = reset_token_id
        self.bidirectional = bidirectional
        self.batch_first = batch_first




class LockedDropout(nn.Module):
    '''
    Dropout for the same inputs at each call
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class WeightDrop(nn.Module):
    """
    from https://github.com/a-martyn/awd-lstm/blob/master/model/net.py
    This only works for the custom lstm cell. If using default LSTM, 
    use this https://github.com/fastai/fastai2/blob/master/fastai2/text/models/awdlstm.py#L29
    Adapted to also handle h2h_reverse weights for bidirecional lstm.
    _____
    A module that wraps an LSTM cell in which some weights will be replaced by 0 during training.
    Adapted from: https://github.com/fastai/fastai/blob/master/fastai/text/models.py
    
    Initially I implemented this by getting the models state_dict attribute, modifying it to drop
    weights, and then loading the modified version with load_state_dict. I had to abandon this 
    approach after identifying it as the source of a slow memory leak.
    """
    #TODO make bidirectional.
    def __init__(self, module:nn.Module, weight_p:float):
        super().__init__()
        self.module, self.weight_p = module, weight_p
        
        self.bidirectional = self.module.bidirectional

        #Makes a copy of the weights of the selected layers.
        w = getattr(self.module.h2h, 'weight')
        self.register_parameter('weight_raw', nn.Parameter(w.data))
        self.module.h2h._parameters['weight'] = F.dropout(w, p=self.weight_p, training=False)

        if self.bidirectional:
            w_rev = getattr(self.module.h2h_reverse, 'weight')
            self.register_parameter('weight_raw_rev', nn.Parameter(w_rev.data))
            self.module.h2h_reverse._parameters['weight'] = F.dropout(w_rev, p=self.weight_p, training=False)


    def _setweights(self):
        "Apply dropout to the raw weights."
        raw_w = getattr(self, 'weight_raw')
        self.module.h2h._parameters['weight'] = F.dropout(raw_w, p=self.weight_p, training=self.training)

        if self.bidirectional:
            raw_w_rev = getattr(self, 'weight_raw_rev')
            self.module.h2h_reverse._parameters['weight'] = F.dropout(raw_w_rev, p=self.weight_p, training=self.training)


    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            #To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        raw_w = getattr(self, 'weight_raw')
        self.module.h2h._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)

        if self.bidirectional:
            raw_w_rev = getattr(self, 'weight_raw_rev')
            self.module.h2h_reverse._parameters[layer] = F.dropout(raw_w_rev, p=self.weight_p, training=False)

        if hasattr(self.module, 'reset'): self.module.reset()



class LSTMCell(nn.Module):
    """
    LSTM cell to support resetting the hidden state when a end of sequence token is encountered.
    Cannot do that with pytorch default LSTM, as sequence steps are not accessible there.
    based on https://github.com/a-martyn/awd-lstm/blob/master/model/net.py
    Assume seq_len first dimension.
    """

    def __init__(self, input_size, output_size, bias=True, dropout =0, bidirectional = False, reset_token_id: int = -10000):
        super(LSTMCell, self).__init__()
        
        # Contains all weights for the 4 linear mappings of the input x
        # e.g. Wi, Wf, Wo, Wc
        self.i2h = nn.Linear(input_size, 4*output_size, bias=bias)
        # Contains all weights for the 4 linear mappings of the hidden state h
        # e.g. Ui, Uf, Uo, Uc
        self.h2h = nn.Linear(output_size, 4*output_size, bias=bias)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.i2h_reverse = nn.Linear(input_size, 4*output_size, bias=bias)
            self.h2h_reverse = nn.Linear(input_size, 4*output_size, bias=bias)
        self.output_size = output_size
        self.reset_token_id = reset_token_id
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.output_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            

    def _cell_step(self, x, hidden, tokens = None):
        '''
        performs one lstm update step.
        tokens: token_ids of the previous step. 
        If previous step had a reset_token, hidden state for this batch element is reset to 0 before the update.
        '''
        # unpack tuple (recurrent activations, recurrent cell state)
        #---- single cell time step operation
        h, c = hidden
        # reset the hidden states when and end of sequence token is encountered
        if tokens != None: #explicit check because of tensor behaviour
            idx = torch.where(tokens == self.reset_token_id)[0] #indices that are a reset_id token
            #print(idx.shape[0])
            h[idx,:] = 0
            c[idx,:] = 0
        # Linear mappings : all four in one vectorised computation
        preact = self.i2h(x) + self.h2h(h)

        # Activations
        i = torch.sigmoid(preact[:, :self.output_size])                      # input gate
        f = torch.sigmoid(preact[:, self.output_size:2*self.output_size])    # forget gate
        g = torch.tanh(preact[:, 3*self.output_size:])                       # cell gate
        o = torch.sigmoid(preact[:, 2*self.output_size:3*self.output_size])  # output gate


        # Cell state computations: 
        # calculates new long term memory state based on proposed updates c_T
        # and input and forget gate states i_t, f_t
        c_t = torch.mul(f, c) + torch.mul(i, g)

        # Output
        h_t = torch.mul(o, torch.tanh(c_t))
        return h_t, c_t

    def _split_hidden_state(self, hidden_state: List[Tuple[torch.tensor, torch.tensor]]) -> Tuple[Tuple, Tuple]:
        '''Split concatenated hidden states for bidirectional model'''
        h_fwd, h_bwd = torch.chunk(hidden_state[0], 2, dim=-1)
        c_fwd, c_bwd = torch.chunk(hidden_state[1], 2, dim=-1)

        return (h_fwd, c_fwd), (h_bwd, c_fwd)

    def forward(self, input: torch.tensor, hidden_state: Tuple[torch.tensor, torch.tensor] = None, input_tokens = None):
        '''
        input: input tensor
        hidden_state: (h_t, c_t) tuple for inital hidden state
        input_tokens: Original input before embedding, used to reset the hidden state on eos tokens
        '''
        
        if self.bidirectional:
            #split the hidden state
            hidden_state, hidden_state_reverse = self._split_hidden_state(hidden_state)
            #TODO problem. output size keeps doubling through the layers.
            # each time, i get input_dim and return the concatenation, 2*output_dim
                    
        output_list = []
        for t in range(input.size(0)):
            inp = input[t,:,:]
            
            h, c = hidden_state
            #squeeze and unsqueeze ops needed to be compatible with default lstm cell
            if input_tokens is not None:
                previous_tokens = (input_tokens[t-1,:] if t>1 else None) 
                h_t, c_t = self._cell_step(inp, (h.squeeze(), c.squeeze()), previous_tokens)
            else:
                h_t, c_t = self._cell_step(inp, (h.squeeze(), c.squeeze()))
            hidden_state = (h_t.unsqueeze(0), c_t.unsqueeze(0)) #set new hidden state
            output_list.append(h_t)

        output = torch.stack(output_list)

        if self.bidirectional:
            input_reverse = torch.flip(input, [0])

            if input_tokens is not None:
                input_tokens_reverse = torch.flip(input_tokens, [0])
             #split hidden states

            output_list = []
            output_list_reverse = []
            for t in range(input.size(0)):
                inp = input[t,:,:]
                inp_reverse = input[t,:,:]

                h,c = hidden_state
                h_reverse, c_reverse = hidden_state_reverse

                if input_tokens is not None:
                    previous_tokens_reverse = (input_tokens_reverse[t-1,:] if t>1 else None)
                    h_t_reverse, c_t_reverse = self._cell_step(inp_reverse, (h_reverse.squeeze(), c_reverse.squeeze()), previous_tokens_reverse)
                else:
                    h_t_reverse, c_t_reverse = self._cell_step(inp_reverse, (h_reverse.squeeze(), c_reverse.squeeze()))
                hidden_state_reverse = (h_t_reverse.unsqueeze(0), c_t_reverse.unsqueeze(0))
                output_list_reverse.append(h_t_reverse)
            output_reverse = torch.stack(output_list_reverse)

            output = torch.cat([output,torch.flip(output_reverse, [0])], dim =-1) #reverse so positions match, then concat along feature dim
            h_cat = torch.cat([hidden_state[0], hidden_state_reverse[0]], dim =-1)
            c_cat = torch.cat([hidden_state[1], hidden_state_reverse[1]], dim =-1)
            hidden_state = (h_cat, c_cat)

        return output, hidden_state




class ProteinAWDLSTM(nn.Module):
    '''
    Multi-layer AWD-LSTM Model.
    Configuration (pass as `ProteinAWDLSTMConfig` object):
        num_layers:             number of AWD-LSTM layers
        input_size:             size of the inputs to the model (in most cases, size of the embedding layer). Also size of the output.
        hidden_size:            hidden size of the LSTM
        hidden_dropput_prob:    Dropout applied to the output between the LSTM layers
        input_dropout_prob:     Dropout applied to the input before passing through LSTM layers
        dropout_prob:           Dropout applied to final output after LSTM layers
        bidirectional:          Make the LSTM layers bidirectional. Output of model will be 2*input_size, forward and reverse hidden states concatenated.
    Args:
        config:                 ProteinAWDLSTMConfig object
        is_LM:                  Flag to also return outputs without hidden_dropout applied to them. Needed in LM for activation regularization.
    '''
    def __init__(self, config, is_LM: bool):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.input_dropout_prob = config.input_dropout_prob
        self.dropout_prob = config.dropout_prob #for consistency with original, output dropout would be more fitting
        self.bidirectional = config.bidirectional
        self.locked_dropout = LockedDropout() #Same instance reused everywhere
        self.is_LM = is_LM
        self.type_2 = False #this changes the bidirectional LSTM implementation. Type 2 only works with masked LM.

        #setup LSTM cells
        #lstm = [torch.nn.LSTM(config.input_size if l == 0 else config.hidden_size, config.hidden_size if l != self.num_layers - 1 else config.input_size, 1, dropout=0) for l in range(self.num_layers)]
        if self.bidirectional and self.type_2: # this is type2 biLSTM, where the outputs are concatenated between the layers. Does not work for LM.
            lstm = [LSTMCell(config.input_size if l == 0 else config.hidden_size *2, 
                            config.hidden_size if l != self.num_layers - 1 else config.input_size, 
                            1, 
                            dropout=0, 
                            bidirectional= config.bidirectional, 
                            reset_token_id= config.reset_token_id) 
                    for l in range(self.num_layers)]
            if config.weight_dropout_prob:
                lstm = [WeightDrop(layer, config.weight_dropout_prob) for layer in lstm]
                self.lstm = nn.ModuleList(lstm)
        elif self.bidirectional: #type 1 bidirectionality, two separate stacks of LSTMs
            lstm = [LSTMCell(config.input_size if l == 0 else config.hidden_size, 
                config.hidden_size if l != self.num_layers - 1 else config.input_size, 
                1, 
                dropout=0, 
                reset_token_id= config.reset_token_id) 
                for l in range(self.num_layers)]
            lstm_rev = [LSTMCell(config.input_size if l == 0 else config.hidden_size, 
                config.hidden_size if l != self.num_layers - 1 else config.input_size, 
                1, 
                dropout=0, 
                reset_token_id= config.reset_token_id) 
                for l in range(self.num_layers)]
            if config.weight_dropout_prob:
                lstm = [WeightDrop(layer, config.weight_dropout_prob) for layer in lstm]
                self.lstm = nn.ModuleList(lstm)
                lstm_rev = [WeightDrop(layer, config.weight_dropout_prob) for layer in lstm_rev]
                self.lstm_rev = nn.ModuleList(lstm_rev)
        else: 
            lstm = [LSTMCell(config.input_size if l == 0 else config.hidden_size, 
                    config.hidden_size if l != self.num_layers - 1 else config.input_size, 1, dropout=0, 
                    reset_token_id= config.reset_token_id) for l in range(self.num_layers)]
            if config.weight_dropout_prob:
                lstm = [WeightDrop(layer, config.weight_dropout_prob) for layer in lstm]
            self.lstm = nn.ModuleList(lstm)

    def forward(self, inputs, mask = None, hidden_state: List[Tuple[torch.Tensor, torch.Tensor]] = None, input_ids = None):
        '''
        inputs: (seq_len x batch_size x embedding_size)
        hidden_state: output from previous forward pass
        input_ids: original token ids to reset the hidden state

        returns:
            last layer output (all in format of default pytorch lstm)
            all layers hidden states (list)
            all layer outputs before droupout
            all layer outputs after dropout
        '''
        if  hidden_state is None:
            hidden_state = self._init_hidden(inputs.size(1))
        if self.bidirectional and not self.type2:
            h_fwd, h_bwd = torch.chunk(hidden_state[0], 2, dim=-1)
            c_fwd, c_bwd = torch.chunk(hidden_state[1], 2, dim=-1)
            hidden_state, hidden_state_rev = (h_fwd, c_fwd), (h_bwd, c_bwd)

        outputs_before_dropout = []
        hidden_states = []

        inputs = self.locked_dropout(inputs, self.input_dropout_prob)

        #Use last LSTM layer only for LM, for other task i want full size hidden state
        for i, layer in (enumerate(self.lstm) if self.is_LM else enumerate(self.lstm[:-1])):
            #print('hidden state feed to lstm')
            #print(hidden_state[i][0].shape)
            output, new_hidden_state = layer(inputs, hidden_state[i], input_ids)
            #print('return hidden state shape')
            #print(new_hidden_state[0].shape)
            outputs_before_dropout.append(output)
            hidden_states.append(new_hidden_state)
            #apply dropout to hidden states
            if i != (self.num_layers if self.is_LM else self.num_layers-1 ):
                output = self.locked_dropout(output, self.hidden_dropout_prob)

            inputs = output
        
        #apply dropout to last layer output
        output = self.locked_dropout(output, self.dropout_prob)

        if self.bidirectional and not self.type2:
            outputs_before_dropout_rev = []
            hidden_states_rev = []
            inputs_rev = torch.flip(inputs, [0])
            if input_tokens is not None:
                input_tokens_reverse = torch.flip(input_tokens, [0])
            
            for i, layer in enumerate(self.lstm_rev):
                output_rev, new_hidden_state_rev = layer(inputs_rev, hidden_state_rev[i], input_ids_rev)
                outputs_before_dropout_rev.append(output_rev)
                hidden_states_rev.append(new_hidden_state_rev)
                if i != (self.num_layers if self.is_LM else self.num_layers-1 ):
                    output = self.locked_dropout(output_rev, self.hidden_dropout_prob)
                inputs_rev = output_rev

            #concatenate all the forward and backward outputs and states.
            output= torch.cat([output, output_rev], dim = -1)
            outputs_before_dropout = [torch.cat([fwd,rev], dim =-1) for fwd, rev in zip(outputs_before_dropout, outputs_before_dropout_rev)]
            hidden_states = [(torch.cat([h, h_rev], dim =-1), torch.cat([c, c_rev], dim =-1)) for (h, c),(h_rev, c_rev) in zip(hidden_states, hidden_states_rev)]
            #hidden_states : List of tuples

        return output, hidden_states, outputs_before_dropout
    
    def _init_hidden(self, batch_size):
        '''
        Create initial all zero hidden states for the lstm layers
        '''
        weight = next(self.parameters()) #to get the right tensor type
        states = [(weight.new_zeros(1, batch_size, self.hidden_size if l != self.num_layers - 1 else self.input_size),
                    weight.new_zeros(1, batch_size, self.hidden_size if l != self.num_layers - 1 else self.input_size)) for l in range(self.num_layers)]
        if self.bidirectional: # *2 because of concatenation of forward and reverse states
            states = [(weight.new_zeros(1, batch_size, self.hidden_size*2 if l != self.num_layers - 1 else self.input_size*2),
                    weight.new_zeros(1, batch_size, self.hidden_size*2 if l != self.num_layers - 1 else self.input_size*2)) for l in range(self.num_layers)]
        return states




class ProteinAWDLSTMAbstractModel(ProteinModel):
    config_class = ProteinAWDLSTMConfig
    pretrained_model_archive_map = AWDLSTM_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix='awdlstm'

    def _init_weights(self, module):
        #as in Salesforce AWDLSTM
        initrange = 0.1
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.uniform_(-initrange, initrange) #for the embedding
        if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

#embedding model
class ProteinAWDLSTMModel(ProteinAWDLSTMAbstractModel):
    '''
    Model to embed protein sequences.
    '''
    def __init__(self, config: ProteinAWDLSTMConfig, is_LM = False):
        super().__init__(config)
        self.is_LM = is_LM #Flag to control whether the embedding is used for LM, e.g. use last LSTM layer and return more vars
        self.embedding_dropout_prob = config.embedding_dropout_prob
        self.embedding_layer = nn.Embedding(config.vocab_size, config.input_size)
        self.encoder = ProteinAWDLSTM(config, is_LM = is_LM)
        self.output_hidden_states = config.output_hidden_states
        self.reset_token_id = config.reset_token_id
        self.batch_first = config.batch_first

        self.init_weights()

    def forward(self, input_ids, input_mask = None, hidden_state = None):
        '''
        is_LM: Flag to return all layer outputs for regularization
        Returns:
            output: lstm hidden states of last layer of shape (seq_len, batch_size, hidden_size)
            hidden_state: List of (h_x, o_x) tuples of last hidden state for each lstm layer
            (outputs_raw): lstm hidden states of all layers without dropout applied to them, used for activation regularization
        '''
        if self.batch_first:
            input_ids = input_ids.transpose(0,1)
            if input_mask is not None:
                input_mask = input_mask.transpose(0,1)

        if input_mask is None:
            input_mask = torch.ones_like(input_ids)
        # fp16 compatibility
        input_mask = input_mask.to(dtype=next(self.parameters()).dtype)
    
        embedding_output = self._embedded_dropout(embed = self.embedding_layer, words =input_ids, dropout=self.embedding_dropout_prob if self.training else 0)


        if self.reset_token_id is not None:
            encoder_outputs = self.encoder(embedding_output, input_mask, hidden_state, input_ids)
        else:
            encoder_outputs = self.encoder(embedding_output, input_mask, hidden_state)

        output, hidden_state, outputs_raw = encoder_outputs
        #TODO maybe implement some form of pooling here
        if self.batch_first:
            output = output.transpose(0,1) 

        if self.is_LM:
            return output, hidden_state, outputs_raw
        return output, hidden_state

    def _embedded_dropout(self, embed, words, dropout=0.1, scale=None):
        if dropout:
            mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
            masked_embed_weight = mask * embed.weight
        else:
            masked_embed_weight = embed.weight
        if scale:
            masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

        padding_idx = embed.padding_idx
        if padding_idx is None:
            padding_idx = torch.tensor(-1, dtype = words.dtype) #ensure creation on cuda

        X = torch.nn.functional.embedding(words, masked_embed_weight,
            padding_idx, embed.max_norm, embed.norm_type,
            embed.scale_grad_by_freq, embed.sparse
        )
        return X


class ProteinAWDLSTMForLM(ProteinAWDLSTMAbstractModel):
    '''
    Model to run the original AWD-LSTM pretraining strategy.
    - reuse the hidden state
    - activation regularization

    Supports bidirectional training. As we want the loss to be interpretable as perplexity, it
    performs next token prediction seperately for the forward and backward LSTM outputs. This 
    means that also targets_reverse need to be given, as the tokens the backwards model predicts
    are data[train_idx -1: train_idx] as opposed to data[train_idx]
    WAIT A SEC THIS IS ONLY TRUE FOR ANNOYING TBBT TRAINING, IN GENERAL [START-1] DOESNT EVEN EXIST
    '''
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ProteinAWDLSTMModel(config = config, is_LM = True)
        self.decoder = nn.Linear(config.input_size, config.vocab_size)
        #some people say this does not work, but is in original code
        self.decoder.weight = self.encoder.embedding_layer.weight
        self.alpha = config.alpha
        self.beta = config.beta

        self.init_weights()

    def _regularize_loss(self, loss: torch.Tensor, sequence_output: torch.Tensor, last_output: torch.Tensor) -> torch.Tensor:
        if self.alpha:
            ar = self.alpha * sequence_output.pow(2).mean()
            loss += ar
        if self.beta:
             #regularization on the difference between steps, computed for the last layer output only
            #squared difference h_after_step - h_before_step
            tar = self.beta * (last_output[1:] - last_output[:-1]).pow(2).mean()
            loss += tar
        return loss

    def forward(self, input_ids, input_mask=None, hidden_state = None, targets =None):
        outputs = self.encoder(input_ids, input_mask, hidden_state)
        sequence_output, hidden_state, raw_outputs = outputs[:3] #raw_outputs: list of [seq_len x batch_size x output_dim] tensors
        last_layer_raw_output = raw_outputs[-1]
        #seq_output dim 0 is controlled by batch_first
        #raw_outputs is always seq_len_first

        sequence_output_full = sequence_output #for regularization
        last_layer_raw_output_full = last_layer_raw_output #for regularization
        
        if self.encoder.encoder.bidirectional:
            #if the LM is bidirectional, decoding is done twice.
            sequence_output, sequence_output_reverse = torch.chunk(sequence_output,2, dim =-1)
            last_layer_raw_output, last_layer_raw_output_reverse = torch.chunk(last_layer_raw_output, 2, dim = -1)
            prediction_scores = self.decoder(sequence_output)
            prediction_scores_reverse = self.decoder(sequence_output_reverse)
            outputs = torch.cat([prediction_scores, prediction_scores_reverse], dim =-1), hidden_state
        else:
            prediction_scores = self.decoder(sequence_output)
            outputs = prediction_scores, hidden_state

        prediction_scores = prediction_scores.contiguous()

        if targets is not None:
            targets_fwd = targets[1:]
            scores_fwd = prediction_scores[:-1]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            raw_loss = loss_fct(
                scores_fwd.view(-1, self.config.vocab_size), targets_fwd.view(-1))
            
            
            if self.encoder.encoder.bidirectional:
                targets_rev = targets[:-1]
                scores_rev = prediction_scores_reverse[1:]
                raw_loss_reverse = loss_fct(
                    scores_rev.view(-1, self.config.vocab_size), targets_rev.view(-1))


            raw_loss = (raw_loss + raw_loss_reverse) /2.
            lm_loss = self._regularize_loss(raw_loss, sequence_output_full, last_layer_raw_output_full)
            outputs = (lm_loss, raw_loss) + outputs
            
        # (lm_loss, raw_loss), prediction_scores, hidden_states
        return outputs


class ProteinAWDLSTMforSPTagging(ProteinAWDLSTMAbstractModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ProteinAWDLSTMModel(config = config, is_LM = False)
        self.tagging_model = CRFSequenceTaggingHead(input_dim = config.hidden_size, num_tags = config.num_labels, constraints = [(1,0)])

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets =None):
        outputs = self.encoder(input_ids, input_mask)
        sequence_output, _ = outputs
        sequence_output = sequence_output.transpose(0,1) #reshape to batch_first
        outputs = self.tagging_model(sequence_output, input_mask = input_mask, targets = targets)
        #loss, marginal_probs, best_path

        return outputs







class ProteinAWDLSTMForSequenceClassification(ProteinAWDLSTMAbstractModel):
    '''Top model for multi-class classification of sequences.
    1. Embed sequences
    2. Pool hidden states
    3. Pooled output to classifier
    '''
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.batch_first = config.batch_first
        self.classifier_hidden_size = config.classifier_hidden_size
        self.lm_hidden_size = config.hidden_size
        self.dropout = 0 #TODO might want to optimize

        self.encoder = ProteinAWDLSTMModel(config = config, is_LM = False)
        self.output_pooler = SimpleMeanPooling(batch_first = self.batch_first)
        self.classifier = nn.Sequential(
            weight_norm(nn.Linear(self.lm_hidden_size, self.classifier_hidden_size), dim=None),
            nn.ReLU(),
            nn.Dropout(self.dropout, inplace=True),
            weight_norm(nn.Linear(self.classifier_hidden_size, self.num_labels), dim=None))


    def forward(self, input_ids, input_mask = None, targets = None):

        output, hidden_state = self.encoder(input_ids, input_mask)
        
        assert output.shape[0] == targets.shape[0], f"LM output dim 0 ({ output.shape[0]}) and targets dim 0 ({ targets.shape[0]}) do not match. Check batch_first flag in config."

        pooled_outputs = self.output_pooler(output, input_mask)
        logits = self.classifier(pooled_outputs)
        probs = F.softmax(logits, dim = -1)
        outputs = (probs,)

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits, targets)
            outputs = (classification_loss,) + outputs

        return outputs