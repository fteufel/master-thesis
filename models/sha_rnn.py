#SHA-RNN implementation from Merrity's repo
#Refactored to a Huggingface-like API, based on TAPE
#Felix June 2020
#TODO currently does not support hidden state resetting on EOS tokens (relevant when training with concatenated sequences.)
#Not straightforward to implement, in addition to resetting the hidden state would also need to apply attention mask to prevent information flow.


import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
import logging
import warnings
import math
from torch.nn.utils import weight_norm

#Felix 29062020 FusedLayerNorm not available on hpc, failed to install apex from source
#if torch.cuda.is_available():
#    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
#else:
from torch.nn import LayerNorm

from tape.models.modeling_utils import ProteinConfig, ProteinModel

logger = logging.getLogger(__name__)

#This is just here for compatibility, otherwise useless
URL_PREFIX = "https://s3.amazonaws.com/proteindata/pytorch-models/"
SHARNN_PRETRAINED_CONFIG_ARCHIVE_MAP: typing.Dict[str, str] = {}
SHARNN_PRETRAINED_MODEL_ARCHIVE_MAP: typing.Dict[str, str] = {}

#This seems to be some memory-related trick. Unclear how this behaves with pytorch
checkpoint = lambda f, *args, **kwargs: f(*args, **kwargs)



class ProteinSHARNNConfig(ProteinConfig):
    pretrained_config_archive_map = SHARNN_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size: int = 30,
                 input_size: int = 400,
                 hidden_size: int = 1150,
                 num_hidden_layers: int = 3,
                 num_max_positions: int = 2000,
                 dropout_prob: float = 0.4,
                 hidden_dropout_prob: float = 0.3,
                 embedding_dropout_prob: float = 0.1,
                 input_dropout_prob: float = 0.65,
                 weight_dropout_prob: float = 0.5,
                 layer_norm_eps: float = 1e-12,
                 initializer_range: float = 0.02,
                 beta: float = 1 ,
                 alpha: float = 2,
                 reset_token_id: int = -1000,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.input_size = input_size #is the embedding layer size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_max_positions = num_max_positions
        self.dropout_prob = dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.embedding_dropout_prob = embedding_dropout_prob
        self.input_dropout_prob = input_dropout_prob
        self.weight_dropout_prob = weight_dropout_prob
        self.alpha = alpha
        self.beta = beta
        self.reset_token_id = reset_token_id

        self.num_heads = 1


## Utils

class LSTMCell(nn.Module):
    """
    LSTM cell to support resetting the hidden state when a end of sequence token is encountered.
    Cannot do that with pytorch default LSTM, as sequence steps are not accessible there.
    based on https://github.com/a-martyn/awd-lstm/blob/master/model/net.py
    Assume seq_len first dimension.
    """

    def __init__(self, input_size, output_size, bias=True, dropout =0, reset_token_id: int = -10000):
        super(LSTMCell, self).__init__()
        
        # Contains all weights for the 4 linear mappings of the input x
        # e.g. Wi, Wf, Wo, Wc
        self.i2h = nn.Linear(input_size, 4*output_size, bias=bias)
        # Contains all weights for the 4 linear mappings of the hidden state h
        # e.g. Ui, Uf, Uo, Uc
        self.h2h = nn.Linear(output_size, 4*output_size, bias=bias)
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

    def forward(self, input: torch.tensor, hidden_state: typing.Tuple[torch.tensor, torch.tensor] = None, input_tokens = None):
        '''
        input: input tensor
        hidden_state: (h_t, c_t) tuple for inital hidden state
        input_tokens: Original input before embedding, used to reset the hidden state on eos tokens
        '''

        output_list = []
        for t in range(input.size(0)):
            inp = input[t,:,:]
            
            h, c = hidden_state
            #squeeze and unsqueeze ops needed to be compatible with default lstm cell
            if input_tokens != None:
                previous_tokens = (input_tokens[t-1,:] if t>1 else None) 
                h_t, c_t = self._cell_step(inp, (h.squeeze(), c.squeeze()), previous_tokens)
            else:
                h_t, c_t = self._cell_step(inp, (h.squeeze(), c.squeeze()))
            hidden_state = (h_t.unsqueeze(0), c_t.unsqueeze(0)) #set new hidden state
            output_list.append(h_t)

        output = torch.stack(output_list)
        return output, hidden_state

def attention(query, key, value, attn_mask=None, need_weights=True, dropout=None):
    # https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html
    # Needs [batch, heads, seqlen, hid]

    batch_size, heads, query_len, dim = query.size()
    key_len = key.size(2)

    # Scaling by dim due to http://nlp.seas.harvard.edu/2018/04/03/attention.html
    attention_scores = torch.matmul(query, key.transpose(-1, -2).contiguous()) / math.sqrt(dim)
    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(1) # give 3d attention mask same shape as attention_scores
        attention_scores = attention_scores + attn_mask # Mask is additive and contains -Infs

    attention_weights = F.softmax(attention_scores, dim=-1)
    if dropout:
        attention_weights = dropout(attention_weights)
    attention_weights = attention_weights.view(batch_size, heads, query_len, key_len)

    mix = torch.matmul(attention_weights, value)
    return mix, attention_weights

class SingleHeadAttention(nn.Module):
    '''Attention module from original repo, minus all the stuff that was not used anymore per default
    '''
    def __init__(self, hidden_size, dropout = None):
        super().__init__()
        self.nhid = hidden_size
        self.qs = nn.Parameter(torch.zeros(size=(1, 1, hidden_size), dtype=torch.float))
        self.ks = nn.Parameter(torch.zeros(size=(1, 1, hidden_size), dtype=torch.float))
        self.vs = nn.Parameter(torch.zeros(size=(1, 1, hidden_size), dtype=torch.float))
        self.overparam = Overparam(hidden_size)
        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.query_ln = LayerNorm(hidden_size, eps=1e-12)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, query, key, value, attn_mask = None, batch_first = False):
        #sigmoid for the learnable vector-vector multiplications
        qs, ks, vs = torch.sigmoid(self.qs), torch.sigmoid(self.ks), torch.sigmoid(self.vs)
        # over-parametrize vs
        vs = self.overparam(vs)
        # linear and layer norm on query
        query = self.query_linear(query)
        query = self.query_ln(query.float())
        #vector-vector ops
        q, k, v = qs * query, ks * key, vs * value
        #dropout
        if self.drop:
            q, k, v = self.drop(q), k, self.drop(v)

        if not batch_first:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        #check for correct sizes
        batch_size, query_len, nhid = q.size()
        assert nhid == self.nhid
        key_len = k.size(1)
        #reshaping stuff
        q = q.view(batch_size, query_len, 1, self.nhid).transpose(1, 2)
        k, v = [vec.view(batch_size, key_len, 1, self.nhid).transpose(1, 2) for vec in [k, v]]

        #get the attention scores
        output, attn_weights = attention(q, k, v, dropout=self.drop, attn_mask=attn_mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.nhid)
        if not batch_first:
            output = output.transpose(0, 1)

        return output, attn_weights
 


class Overparam(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.l1 = nn.Linear(nhid, 2 * nhid)
        #self.l2 = nn.Linear(2 * nhid, 2 * nhid)
        self.inner_act = torch.tanh # GELU()
        self.nhid = nhid

    def forward(self, x):
        c, f = self.l1(x).split(self.nhid, dim=-1)
        #c, f = self.l2(self.inner_act(self.l1(x))).split(self.nhid, dim=-1)
        return torch.sigmoid(f) * torch.tanh(c)


class Boom(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, shortcut=False):
        super(Boom, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) if dropout else None
        if not shortcut:
            self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.shortcut = shortcut
        self.act = GELU()

    def forward(self, input):
        x = self.act(self.linear1(input))
        if self.dropout: x = self.dropout(x)
        if self.shortcut:
            # Trim the end off if the size is different
            ninp = input.shape[-1]
            x = torch.narrow(x, -1, 0, x.shape[-1] // ninp * ninp)
            # Divide the hidden size evenly into chunks
            x = x.view(*x.shape[:-1], x.shape[-1] // ninp, ninp)
            # Collapse the chunks through summation
            z = x.sum(dim=-2)
        else:
            z = self.linear2(x)

        return z

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        #return torch.nn.functional.gelu(x.float())
        # The first approximation has more operations than the second
        # See https://arxiv.org/abs/1606.08415
        #return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return x * torch.sigmoid(1.702 * x)


# Models

class SHARNN(nn.Module):
    '''
    Single layer SHA-RNN (called a block in the original repo)
    Supports attention masking and hidden state resetting.
    For this, needs input_ids, input_ids_mem - look for eos tokens, reset hidden state after this and mask attention positions for following seqs
    '''
    def __init__(self, embed_dim, hidden_dim, heads=1, dropout=None, use_attn=True, num_max_positions =2000, reset_token_id: int = -10000):
        super().__init__()
        
        if use_attn:
            self.attn = SingleHeadAttention(embed_dim, dropout=dropout)
        else:
            self.attn = None

        self.ff = Boom(embed_dim, embed_dim, dropout=dropout, shortcut=True)
        self.lnstart = LayerNorm(embed_dim, eps=1e-12)
        self.lnmid = LayerNorm(embed_dim, eps=1e-12)
        self.lnmem = LayerNorm(embed_dim, eps=1e-12)
        self.lnout = LayerNorm(embed_dim, eps=1e-12)
        self.lnff = LayerNorm(embed_dim, eps=1e-12)
        self.lnxff = LayerNorm(embed_dim, eps=1e-12)
        self.drop = nn.Dropout(dropout)
        self.gelu = GELU()

        self.rnn = LSTMCell(input_size = embed_dim, output_size = embed_dim, dropout=0,reset_token_id = reset_token_id)
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=False)
        self.num_max_positions = num_max_positions
        self.reset_token_id = reset_token_id


    def forward(self, inputs, attn_mask, mem=None, hidden=None, input_ids =  None, input_ids_mem = None):
        '''
        Paper:
        Input
          V
        LayerNorm
          V
        LSTM ---------------< hidden
          |-----------------< Memory
          V           V   V
        LayerNorm     Q  LayerNorm
          |           V   V
          |--------->Attention
         Sum<------------|
          |---->Boom
          V      |
         Sum <---|
          |
          V
        inputs = h
        '''
        new_mem = None #for when we don't use attention
        new_id_mem = None


        inputs = self.lnstart(inputs)

        rnn_output, new_hidden = self.rnn(inputs, None if hidden is None else hidden)
        # Trim the end off if the size is different
        ninp = inputs.shape[-1]
        #z = torch.narrow(rnn_output, -1, 0, rnn_output.shape[-1] // ninp * ninp) #probably remnant, overwritten by next z =
        # Divide the hidden size evenly into chunks
        z = rnn_output.view(*rnn_output.shape[:-1], rnn_output.shape[-1] // ninp, ninp)
        # Collapse the chunks through summation
        rnn_output = self.drop(z).sum(dim=-2)

        #Attention with Memory state - current sequence output gets attention to full history window
        focus, new_mem = None, []
        
        mh = self.lnmem(rnn_output)
        h = self.lnmid(rnn_output)

        #add new hidden states to memory
        if mem is not None:
            bigh = torch.cat([mem, mh], dim=0)
        else:
            bigh = mh
        #drop oldest hidden states from memory to retain a history window of num_max_positions
        new_mem = bigh[-self.num_max_positions:] #get max_seq_len last elements. e.g. dim 5000, max_seq_len 1000 , get 4000-5000

        # make a new attention mask to add
        #Attention function supports 3D attention mask tensors, with batch size as first dimension
        

        #Attention
        if self.attn is not None:
            q, k = h, bigh
            attn_mask = attn_mask.unsqueeze(0) #TODO use while real attention mask is not 3D
            attn_mask = attn_mask.repeat(input_ids.shape[-1],1,1) #tile mask along batch dim

            #Mask the attention mask to treat each real sequence in the concatenated sequence individually
            if input_ids is not None: #fix attention mask for eos tokens
                #update id memory
                if input_ids_mem is not None:
                    bigids = torch.cat([input_ids_mem, input_ids], dim=0)
                else:
                    bigids = input_ids
                
                new_id_mem = bigids[-self.num_max_positions:]

                for i in range(attn_mask.shape[0]): #iterate over batch dims
                    #get positions on which to reset
                    seq_idx = torch.where(input_ids[:,i] == self.reset_token_id)[0] #after these positions attention needs to be masked
                    # make mask -Inf blockwise for each eos idx
                    for idx in seq_idx:
                        # attn mask is (rnn_output_len, memory_window_len) -> offset mem.shape[0] to edit attentions between seq positions
                        attn_mask[i,idx+1:,:idx+1+mem.shape[0]] = -float('Inf')

                    #also mask memory
                    if input_ids_mem is not None: #skip at first run, no memory yet
                        mem_idx = torch.where(input_ids_mem[:,i] == self.reset_token_id)[0]
                        if len(mem_idx)>0 :
                            last_mem_idx = mem_idx[-1]
                            attn_mask[i,:,:last_mem_idx+1] = -float('Inf')


            attention_output, focus = checkpoint(self.attn, q, k, bigh, attn_mask)
            attention_output = self.drop(attention_output)
            output = attention_output + rnn_output
        else:
            output = rnn_output


        #Boom
        output, boom_input = self.lnff(output), self.lnxff(output)
        boom_output = checkpoint(self.ff, boom_input)
        boom_output = self.drop(boom_output)
        output = boom_output + output


        return output, new_mem, new_hidden, new_id_mem, focus


class SHARNNModel(nn.Module):
    '''
    Multi-Layer SHA-RNN Model.
    Applies dropouts, processes memory and creates attention mask.
    NOTE: Does not used locked dropout and weight dropout as opposed to awd-lstm

    '''
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.num_max_positions = config.num_max_positions
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.input_dropout_prob = config.input_dropout_prob
        self.dropout_prob = config.dropout_prob #for consistency with original, output dropout would be more fitting

        #not in config for some reason
        self.num_heads =1

        self.drop = nn.Dropout(config.dropout_prob)
        self.idrop = nn.Dropout(config.input_dropout_prob)
        self.hdrop = nn.Dropout(config.hidden_dropout_prob)

        #TODO attention heads hardcoded. removing constraint doubles epoch time and gives a bit extra
        #https://github.com/Smerity/sha-rnn/issues/3
        layers = [SHARNN(self.input_size, self.hidden_size, self.num_heads, dropout=self.hidden_dropout_prob, num_max_positions=self.num_max_positions, reset_token_id=config.reset_token_id, use_attn=True if l == self.num_layers - 2 else False) for l in range(self.num_layers) ]

        self.layers = nn.ModuleList(layers)

    def forward(self, inputs, mask = None, hidden_state = None, memory = None, input_ids = None, id_memory = None):
        '''hidden state passing works the same as in AWD-LSTM.
        So i get a size n_layer tuple of (h_x, c_x) tuples. Additionally, also memory tuple (only 1 tensor per layer)
        '''

        inputs = self.idrop(inputs)
        if memory is not None:
            maxmem = self.num_max_positions - len(inputs) #len(tensor) gives tensor.shape[0], works fine
            memory = [m[-maxmem:] for m in memory]

        total_length = len(inputs) + (len(memory[0]) if memory else 0) 


        new_hiddens = []
        new_mems = []
        new_id_mems = []

        attn_mask = torch.full((len(inputs), len(inputs)), -float('Inf'), device=inputs.device, dtype=inputs.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        #Attention mask is causal for current sequence. Need to cat with 0 for the memory that also goes into attention
        if memory:
            max_mems = max(len(m) for m in memory)
            happy = torch.zeros((len(inputs), max_mems), device=inputs.device, dtype=inputs.dtype)
            attn_mask = torch.cat([happy, attn_mask], dim=-1)

        output = inputs
        for l, layer in enumerate(self.layers):
            mem = memory[l] if memory else None
            hid = hidden_state[l] if hidden_state else None #select correct tuple
            id_mem = id_memory[l] if id_memory else None

            output, new_mem, new_hidden, new_id_mem, focus = layer(output, attn_mask, mem = mem, hidden = hid, input_ids = input_ids, input_ids_mem = id_mem)

            new_hiddens.append(new_hidden)
            new_mems.append(new_mem)
            new_id_mems.append(new_id_mem)
            output

        output = self.drop(output)

        return output, new_hiddens, new_mems, new_id_mems


class ProteinSHARNNAbstractModel(ProteinModel):
    config_class = ProteinSHARNNConfig
    pretrained_model_archive_map = SHARNN_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "sharnn"



    def _init_weights(self, module):
        '''
        Note that this will fail when the child class that calls self.init_weights() does not implement self.input_size
        '''
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=0.1 / math.sqrt(self.input_size)) 

        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()


#Embedding model
class ProteinSHARNNModel(ProteinSHARNNAbstractModel):
    def __init__(self, config: ProteinSHARNNConfig):
        super().__init__(config)
        
        self.embedding_dropout_prob = config.embedding_dropout_prob
        self.embedding_layer = nn.Embedding(config.vocab_size, config.input_size)
        self.encoder = SHARNNModel(config)
        self.output_hidden_states = config.output_hidden_states
        self.input_size = config.input_size
        self.init_weights()



    def forward(self, input_ids, input_mask = None, hidden_state = None, memory = None, id_memory = None):
        '''
        Takes tokenized input data (token IDs tensor)
        returns output, last hidden LSTM state and memory
        '''

        if input_mask is None:
            input_mask = torch.ones_like(input_ids)
        # fp16 compatibility
        input_mask = input_mask.to(dtype=next(self.parameters()).dtype)
    
        embedding_output = self._embedded_dropout(embed = self.embedding_layer, words =input_ids, dropout=self.embedding_dropout_prob if self.training else 0)

        encoder_outputs = self.encoder(embedding_output, input_mask, hidden_state, memory, input_ids, id_memory) #output, hidden_states, memory
    
        output, hidden_state, memory, id_memory = encoder_outputs

        return output, hidden_state, memory, id_memory



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
            padding_idx = -1

        X = torch.nn.functional.embedding(words, masked_embed_weight,
            padding_idx, embed.max_norm, embed.norm_type,
            embed.scale_grad_by_freq, embed.sparse)
        return X
        

class ProteinSHARNNForLM(ProteinSHARNNAbstractModel):
    '''
    '''
    def __init__(self, config):
        super().__init__(config)
        self.input_size = config.input_size
        self.encoder = ProteinSHARNNModel(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        #some people say this does not work, but is in original code
        self.decoder.weight = self.encoder.embedding_layer.weight


        self.init_weights()

    def forward(self, input_ids, input_mask=None, hidden_state = None, memory = None, id_memory = None, targets =None):
        outputs = self.encoder(input_ids, input_mask, hidden_state, memory, id_memory)
        sequence_output, hidden_state, memory, id_memory = outputs[:4]

        prediction_scores = self.decoder(sequence_output)
        outputs = prediction_scores, hidden_state, memory, id_memory


        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), targets.view(-1))
              
            outputs = (lm_loss,) + outputs
            
        # (loss), prediction_scores, hidden_states
        return outputs