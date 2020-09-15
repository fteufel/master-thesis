'''
Convert weights from an AWD-LSTM model in my own implementation so that they can be loaded in fastai.
'''
import torch
import argparse
import os
import fastai.text
import json

#maps from fastai to tape.
WEIGHT_NAME_MAP = {
                    'encoder.weight':             'encoder.embedding_layer.weight',
                    'encoder_dp.emb.weight':      'encoder.embedding_layer.weight', #my implemenation uses functional dropout for this, no module. weights are the same.
                    'rnns.0.weight_hh_l0_raw':    'encoder.encoder.lstm.0.weight_raw', 
                    'rnns.0.module.weight_ih_l0': 'encoder.encoder.lstm.0.module.i2h.weight', 
                    'rnns.0.module.weight_hh_l0': 'encoder.encoder.lstm.0.module.h2h.weight', 
                    'rnns.0.module.bias_ih_l0' :  'encoder.encoder.lstm.0.module.i2h.bias', 
                    'rnns.0.module.bias_hh_l0':   'encoder.encoder.lstm.0.module.h2h.bias', 
                    'rnns.1.weight_hh_l0_raw':    'encoder.encoder.lstm.1.weight_raw', 
                    'rnns.1.module.weight_ih_l0': 'encoder.encoder.lstm.1.module.i2h.weight',
                    'rnns.1.module.weight_hh_l0': 'encoder.encoder.lstm.1.module.h2h.weight',
                    'rnns.1.module.bias_ih_l0':   'encoder.encoder.lstm.1.module.i2h.bias',
                    'rnns.1.module.bias_hh_l0':   'encoder.encoder.lstm.1.module.h2h.bias',
                    'rnns.2.weight_hh_l0_raw':    'encoder.encoder.lstm.2.weight_raw',
                    'rnns.2.module.weight_ih_l0': 'encoder.encoder.lstm.2.module.i2h.weight', 
                    'rnns.2.module.weight_hh_l0': 'encoder.encoder.lstm.2.module.h2h.weight', 
                    'rnns.2.module.bias_ih_l0':   'encoder.encoder.lstm.2.module.i2h.bias', 
                    'rnns.2.module.bias_hh_l0':   'encoder.encoder.lstm.2.module.h2h.bias', 
                    'rnns.3.weight_hh_l0_raw':    'encoder.encoder.lstm.3.weight_raw',
                    'rnns.3.module.weight_ih_l0': 'encoder.encoder.lstm.3.module.i2h.weight', 
                    'rnns.3.module.weight_hh_l0': 'encoder.encoder.lstm.3.module.h2h.weight', 
                    'rnns.3.module.bias_ih_l0':   'encoder.encoder.lstm.3.module.i2h.bias', 
                    'rnns.3.module.bias_hh_l0':   'encoder.encoder.lstm.3.module.h2h.bias',
                    '1.decoder.weight':             'decoder.weight',
                    '1.decoder.bias':               'decoder.bias',
                    }




parser = argparse.ArgumentParser(description='convert weights from tape to fastai')
parser.add_argument('--checkpoint', type = str, default = '/work3/felteu/sp_tagging_checkpoints/xlnet')
parser.add_argument('--output_dir', type = str, default = 'converted_checkpoints')
args = parser.parse_args()


# load config and get dimensions
config = json.load(open(os.path.join(args.checkpoint, 'config.json'), 'r'))

lstm_model = fastai.text.AWD_LSTM(vocab_sz = config['vocab_size'],
                                 emb_sz = config['input_size'],
                                 n_hid = config['hidden_size'],
                                 n_layers = config['num_hidden_layers'],
                                 pad_token =0,
                                 hidden_p = config['hidden_dropout_prob'],
                                 input_p = config['input_dropout_prob'],
                                 embed_p = config['embedding_dropout_prob'],
                                 weight_p = config['weight_dropout_prob'] ,
                                 )

state_dict = lstm_model.state_dict()

tape_checkpoint = torch.load(os.path.join(args.checkpoint,'pytorch_model.bin'), map_location=torch.device('cpu'))

for key in state_dict.keys():
    tape_key = WEIGHT_NAME_MAP[key]
    corresponding_weight = tape_checkpoint[tape_key]
    state_dict[key] = corresponding_weight


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

torch.save(state_dict, os.path.join(args.output_dir, 'fastai_checkpoint.pth'))
