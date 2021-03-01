import torch 
from single_model_for_jit import BertSequenceTaggingCRF
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',type=str,default='/work3/felteu/tagging_checkpoints/signalp_6_distilled/')
parser.add_argument('--output_file', type=str, default="/work3/felteu/tagging_checkpoints/distilled_scripted_list.pt")
args = parser.parse_args()


dummy_input_ids = torch.ones(1, 73, dtype=int)
dummy_input_mask = torch.ones(1, 73, dtype=int)

model = BertSequenceTaggingCRF.from_pretrained(args.checkpoint)
model.eval()

model(dummy_input_ids, dummy_input_mask)
print('Standard mode fwd pass done.')

print('Exporting')
model = torch.jit.trace(model, (dummy_input_ids, dummy_input_mask))


#dummy_input_ids = torch.ones(50, 73, dtype=int)
#dummy_input_mask = torch.ones(50, 73, dtype=int)
print('Traced model')

os.makedirs(os.path.dirname(args.output_file),exist_ok=True)
model.save(args.output_file)
