'''
Export the model to ONNX to get better performance.
We can use tracing, the only dynamic loop of the model is viterbi decoding.
Marginal prob code runs over the full seq length, including padded positions.

Viterbi decoding will be implemented as its own model, as this model here returns the emissions.
'''

import torch
import os
import sys
import time
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") 


from models.multi_crf_bert import ProteinBertTokenizer
from multi_crf_bert_for_export import BertSequenceTaggingCRF





model = BertSequenceTaggingCRF.from_pretrained('/work3/felteu/tagging_checkpoints/signalp_6/test_0_val_1')
model.eval()
print('loaded model')
start = time.time()

dummy_input_ids =  torch.ones(1,73, dtype=int)
dummy_input_mask =  torch.ones(1,73, dtype=int)

outs =  model(dummy_input_ids,input_mask = dummy_input_mask)

print(f'Test one-sample batch: {time.time()-start} seconds in native pytorch.')


# Export the model
torch.onnx.export(model,               # model being run
                  (dummy_input_ids, dummy_input_mask),                         # model input (or a tuple for multiple inputs)
                  "bert_export_test.onnx",   # where to save the model (can be a file or file-like object)
                  #verbose=True,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input_ids', 'input_mask'],   # the model's input names
                  output_names = ['global_probs','marginal_probs','emissions'], # the model's output names
                  dynamic_axes={'input_ids' : {0 : 'batch_size'},    # variable lenght axes
                                'input_mask' : {0 : 'batch_size'},
                                'global_probs' : {0 : 'batch_size'},
                                'marginal_probs' : {0 : 'batch_size'},
                                'emissions' : {0 : 'batch_size'}})


#import onnx

#onnx_model = onnx.load("bert_export_test.onnx")
#onnx.checker.check_model(onnx_model)