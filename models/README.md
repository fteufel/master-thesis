# Models

All models are built in the design used in TAPE and should be directly compatible.

- `ProteinConfig` child class holds parameters for each model architecture
- `model.from_pretrained()` loads saved models. Argument: path to directory with saved model and saved config.

## AWD-LSTM

AWD-LSTM Model adapted from  
https://github.com/salesforce/awd-lstm-lm
https://github.com/a-martyn/awd-lstm

Changes: 
- If a `reset_token_id` is set (default = `-100`), the hidden states and cell states are reset after encountering the token id in the data. Useful when using long concatenated sequences in training. 


## SHA-RNN
SHA-RNN (more precisely: SHA-LSTM) model adapted from  
https://github.com/Smerity/sha-rnn
Changes:
- `reset_token_id` as in AWD-LSTM for LSTM
- Attention mask for `reset_token_id`. Prevents looking back to tokens that are before a reset position.


## Hyperbolic AWD-LSTM


## Modeling Utils

Head models to be used on top of LMs

`CRFSequenceTaggingHead` : Conditional Random Field

- most probable sequence using viterbi decoding
- marginal probabilities using forward-backward
- loss: crossentropy between true labels and marginal probabilities


# Models SignalP 6.0

Everything before here is not relevant anymore.

`crf_layer.py` holds the multi-tag CRF module

## Bert 
`multi_crf_bert.py` is the new model
`signalp_5.py` is a reimplementation of SignalP5.0, largely based on Silas code, changed to use same api as multi_crf_bert.