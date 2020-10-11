# non-backwards compatible update for checkpoint saving from torch1.4 to torch 1.6
# whole thing runs on torch 1.4 so that I can plugin TAPE models
# TPU training runs on torch 1.6
import torch
import argparse

parser = argparse.ArgumentParser('Convert checkpoint to torch 1.4')
parser.add_argument('checkpoint', type=str)
args = parser.parse_args()
assert torch.__version__ != '1.4.0', "Use environment with torch 1.6"
state_dict = torch.load(args.checkpoint)
torch.save(state_dict, args.checkpoint, _use_new_zipfile_serialization=False)

