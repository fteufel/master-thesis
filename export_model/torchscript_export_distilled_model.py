import torch 
from single_model_for_jit import BertSequenceTaggingCRF





dummy_input_ids = torch.ones(1, 73, dtype=int)
dummy_input_mask = torch.ones(1, 73, dtype=int)

model = BertSequenceTaggingCRF.from_pretrained('/work3/felteu/tagging_checkpoints/signalp_6_distilled/')
model.eval()

model(dummy_input_ids, dummy_input_mask)
print('Standard mode fwd pass done.')

print('Exporting')
model = torch.jit.trace(model, (dummy_input_ids, dummy_input_mask))


import IPython
dummy_input_ids = torch.ones(50, 73, dtype=int)
dummy_input_mask = torch.ones(50, 73, dtype=int)
#IPython.embed()

model.save("/work3/felteu/tagging_checkpoints/distilled_scripted_list.pt")