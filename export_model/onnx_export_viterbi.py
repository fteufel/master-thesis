'''
Export Viterbi decoding to ONNX.
Load all checkpoints, average the weights, load these weights into the CRF,
compile and save.
'''

import torch
import os
import sys
import time
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") 


from models.multi_crf_bert import ProteinBertTokenizer
from transformers import BertConfig
from multi_crf_bert_for_export import BertSequenceTaggingCRF


def get_averaged_crf(model_checkpoint_list):
    '''Average weights of a list of model checkpoints,
    then run viterbi decoding on provided emissions.
    Running this on CPU should be fine, viterbi decoding
    does not use a lot of multiplications. Only one for each timestep.
    '''

    # Gather the CRF weights
    start_transitions = [] 
    transitions = []
    end_transitions = []

    for checkpoint in model_checkpoint_list:

        # load the full model, but with a CRF version that has decode() as fwd()
        config = BertConfig.from_pretrained(checkpoint)
        setattr(config, 'load_viterbi_fwd_crf', True)
        model = BertSequenceTaggingCRF.from_pretrained(checkpoint, config=config)

        start_transitions.append(model.crf.start_transitions.data)
        transitions.append(model.crf.transitions.data)
        end_transitions.append(model.crf.end_transitions.data)

    # Average and set model weights
    start_transitions = torch.stack(start_transitions).mean(dim=0)
    transitions = torch.stack(transitions).mean(dim=0)
    end_transitions = torch.stack(end_transitions).mean(dim=0)

    model.crf.start_transitions.data = start_transitions
    model.crf.transitions.data =  transitions
    model.crf.end_transitions.data = end_transitions

    return model.crf


model_list = ['test_0_val_1']#, 'test_0_val_2',  'test_1_val_0', 'test_1_val_2',  'test_2_val_0', 'test_2_val_1']
base_path = '/work3/felteu/tagging_checkpoints/signalp_6/'

crf = get_averaged_crf([base_path+x  for x in model_list])
crf.eval()
print('loaded model')
start = time.time()

dummy_emissions =  torch.ones(1, 70, 37, dtype=float)
dummy_input_mask =  torch.ones(1,70, dtype=int)

crf.do_transition_constraint() # do this outside of model pass once, not at every pass.

# decode() uses torch.jit version of viterbi
#import ipdb; ipdb.set_trace()
viterbi_paths = crf(emissions=dummy_emissions, mask=dummy_input_mask.byte())
# check if same as default decode implementation
viterbi_paths_old = crf._viterbi_decode(dummy_emissions.transpose(0, 1), dummy_input_mask.transpose(0, 1).byte())

#cannot use assert like this - new model makes tensors not lists of ints
#assert viterbi_paths == viterbi_paths_old, 'broke viterbi implemenation. Output no longer the same.'

print(f'Test one-sample batch: {time.time()-start} seconds in native pytorch.')


# Export the model
torch.onnx.export(crf,               # model being run
                  (dummy_emissions, dummy_input_mask.byte()),                         # model input (or a tuple for multiple inputs)
                  "averaged_crf_exported.onnx",   # where to save the model (can be a file or file-like object)
                  #verbose=True,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  verbose=True,
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input_ids', 'input_mask'],   # the model's input names
                  output_names = ['viterbi_path'], # the model's output names
                  dynamic_axes={'input_ids' : {0 : 'batch_size'},    # variable lenght axes
                                'input_mask' : {0 : 'batch_size'},
                                'viterbi_path' : {0 : 'batch_size'}})


