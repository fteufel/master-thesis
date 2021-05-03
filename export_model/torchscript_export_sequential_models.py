'''
Script to export sequential models.
From each checkpoint, produce a model that returns probs, marginal probs and emissions.
Make one pooled CRF decoder model and save.
'''
import torch 
from single_model_for_jit_for_sequential import BertSequenceTaggingCRF, PooledCRF

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir',type=str,default='/work3/felteu/tagging_checkpoints/signalp_6/')
parser.add_argument('--output_dir', type=str, default="/work3/felteu/tagging_checkpoints/torchscript_sequential/")
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

device = torch.device(args.device)
dummy_input_ids = torch.ones(1, 73, dtype=int).to(device)
dummy_input_mask = torch.ones(1, 73, dtype=int).to(device)

os.makedirs(args.output_dir,exist_ok=True)


start_transitions = []
transitions = []
end_transitions = []

# Export the individual models.
for ckp in ['test_0_val_1', 'test_0_val_2', 'test_1_val_0', 'test_1_val_2', 'test_2_val_0', 'test_2_val_1']:

    model = BertSequenceTaggingCRF.from_pretrained(os.path.join(args.checkpoint_dir, ckp))
    model.to(device)
    print('Loaded model.')
    model.eval()

    start_transitions.append(model.crf.start_transitions)
    transitions.append(model.crf.transitions)
    end_transitions.append(model.crf.end_transitions)

    model(dummy_input_ids, dummy_input_mask)
    print('Standard mode fwd pass done.')

    print('Exporting')
    model = torch.jit.trace(model, (dummy_input_ids, dummy_input_mask))
    print('Traced model')

    model.save(os.path.join(args.output_dir, ckp+'.pt'))


# Export the pooled CRF. CRF always runs on CPU, dynamic programming does not gain speed from GPU.
print('Instantiating CRF.')

        
#get CRF weights from berts and average
start_transitions = torch.stack(start_transitions).mean(dim=0).cpu()
transitions = torch.stack(transitions).mean(dim=0).cpu()
end_transitions = torch.stack(end_transitions).mean(dim=0).cpu()

crf = PooledCRF(start_transitions, end_transitions, transitions)
crf.to(torch.device('cpu'))
dummy_emissions = torch.ones(1,70,37).cpu()
#viterbi_paths = self.crf(emissions_mean, input_mask.byte())
crf(dummy_emissions, dummy_input_mask.cpu())
print('Standard mode fwd pass done.')

model = torch.jit.trace(crf, (dummy_emissions, dummy_input_mask.cpu()))
model.save(os.path.join(args.output_dir, 'averaged_viterbi.pt'))
