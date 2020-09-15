'''
Different parameter sets for easy plugging in in train.py.
Useful for debugging training, parameters that work fine on custom model cause nan in fastai.
'''

kwargs_eukarya = {
"cv_fold":-1, #cv-fold -1 for single split else fold
"working_folder":"/work3/felteu/data/", # folder with preprocessed data 
"pretrained_folder":"./lm_sprot", # folder for pretrained model
"model_filename_prefix":"model_debug", # filename for saved model
"model_folder":"",# folder for model to evaluate if train=False (empty string refers to data folder) i.e. model will be located at {model_folder}/models/{model_filename_prefix}_3.pth
"pretrained_model_filename":"model_3_enc", # filename of pretrained model (default for loading a lm encoder); a suffix _enc will load the encoder only otherwise the full model will be loaded

"emb_sz":384, # embedding size
"nh":1088, # number of hidden units
"nl":4, # number of layers
"lin_ftrs":None, #optional list of hidden layer sizes for the classification head (None == [50])

"wd":4.252790725868348e-22, # weight decay
"bptt":32, # backpropagation through time (sequence length fed at a time) in fastai LMs is approximated to a std. deviation around 70, by perturbing the sequence length on a per-batch basis
"max_len":1024, # RNN only- number of tokens for which the loss is backpropagated (last max_len tokens of the sequence) [only for ordinary classification i.e. annotation=False]  see bptt for classification in the paper   BE CAREFUL: HAS TO BE LARGE ENOUGH
"bs":224, # batch size
"dropout":3.806117031547348e-27, # dropout rate
"lr":16.41216994356999, # learning rate
"lr_slice_exponent": 2.6, # learning rate decay per layer group (used if interactive_finegrained=False); set to 1 to disable discriminative lrs
"lr_fixed":False, #do not reduce the lr after each finetuning step
"lr_stage_factor": [2,2,5], #if lr_fixed==False reduce lr after each stage by dividing by this factor (allows to pass a list to specify factor for every stage)
"epochs":30, # epochs for (final) tuning step (pass a list with 4 elements to specify all epochs)
"fp16":False, #fp 16 training
"fit_one_cycle": True, #use one-cycle lr scheduling

#awdlstm params - udsmprot left those at default.
'input_p': 0.65, 
'output_p': 3.806117031547348e-27, 
'weight_p':  4.936400578288149e-08, 
'embed_p': 6.132923880922801e-20, 
'hidden_p': 0.002334712622719755,


"backwards": False, #invert order of all
"clip":0.7033997553965777, # gradient clipping
"train":True, # False for inference only

"from_scratch":True, # train from scratch or pretrain
"gradual_unfreezing":True, # unfreeze layer by layer during finetuning

"hierarchical":None, #list of lists separating different hierarchical losses e.g. [[0.6],[6,-1]] for EC level 1 [0:6] and level 2 [6:-1]

"arch": "AWD_LSTM", # AWD_LSTM, Transformer, TransformerXL, BERT (BERT shares params nh, nl, dropout with the LSTM config)
"nheads":6, # number of BERT/Transformer heads 

"tie_encoder":True, # tie embedding and output for LMs

"max_seq_len":1024, # max. sequence length (required for certain truncation modes and BERT)
"truncation_mode":0, #determines how to deal with long sequences (longer than max_seq_len) 0: keep (default for RNN) 1:keep first max_seq_len tokens (default for BERT) 2:keep last max_seq_len tokens (not consistent with BERT) 3: drop

"eval_on_test":False, #use original test set as validation set (should only be used for train=False or together with concat_train_val)
"eval_on_val_test":False, #use validation set as usual and evaluate on the test set at the end
"concat_train_val":False, #use both train and valid as training set (for fit using all data- has to be used in conjuction with eval_on_test=True)
"metrics":["accuracy"], # array of strings specifying metrics for evaluation (currently supported accuracy, macro_auc, macro_f1, binary_auc, binary_auc50)
"export_preds":False, #outputs validation set predictions as preds.npz (bundled numpy array with entries val_IDs_sorted (validation set IDs corresponding to the entries in IDs.npy), preds (validation set predictions- these are the actual model outputs and might require an additional softmax), targs (the corresponding target labels))

"annotation":False, # for classification only: this is an annotation task (one output per token)
"regression":False, # for classification only: this is a regression task

"early_stopping": "None", #performs early stopping on specified metric (possible entries: valid entries for metrics or trn_loss or val_loss)

"interactive": False, # for execution in juyter environment; allows manual determination of lrs (specifying just the lr for the first finetuning step)
"interactive_finegrained":False, # for execution in juyter environment; allows manual determination of lrs (specifying lrs for all finetuning steps)

"return_learner": False, #returns learner and exits

'wandb_id': 'protein-language-modeling'
}


kwargs_udsmprot = {
"cv_fold":-1, #cv-fold -1 for single split else fold
"working_folder":"./lm_sprot", # folder with preprocessed data 
"pretrained_folder":"./lm_sprot", # folder for pretrained model
"model_filename_prefix":"model", # filename for saved model
"model_folder":"",# folder for model to evaluate if train=False (empty string refers to data folder) i.e. model will be located at {model_folder}/models/{model_filename_prefix}_3.pth
"pretrained_model_filename":"model_3_enc", # filename of pretrained model (default for loading a lm encoder); a suffix _enc will load the encoder only otherwise the full model will be loaded

"emb_sz":400, # embedding size
"nh":1150, # number of hidden units
"nl":3, # number of layers
"lin_ftrs":None, #optional list of hidden layer sizes for the classification head (None == [50])

"wd":1e-7, # weight decay
"bptt":70, # backpropagation through time (sequence length fed at a time) in fastai LMs is approximated to a std. deviation around 70, by perturbing the sequence length on a per-batch basis
"max_len":1024, # RNN only- number of tokens for which the loss is backpropagated (last max_len tokens of the sequence) [only for ordinary classification i.e. annotation=False]  see bptt for classification in the paper   BE CAREFUL: HAS TO BE LARGE ENOUGH
"bs":128, # batch size
"dropout":0.5, # dropout rate
"lr":1e-3, # learning rate
"lr_slice_exponent": 2.6, # learning rate decay per layer group (used if interactive_finegrained=False); set to 1 to disable discriminative lrs
"lr_fixed":False, #do not reduce the lr after each finetuning step
"lr_stage_factor": [2,2,5], #if lr_fixed==False reduce lr after each stage by dividing by this factor (allows to pass a list to specify factor for every stage)
"epochs":30, # epochs for (final) tuning step (pass a list with 4 elements to specify all epochs)
"fp16":False, #fp 16 training
"fit_one_cycle": True, #use one-cycle lr scheduling

"backwards": False, #invert order of all
"clip":.25, # gradient clipping
"train":True, # False for inference only

"from_scratch":True, # train from scratch or pretrain
"gradual_unfreezing":True, # unfreeze layer by layer during finetuning

"hierarchical":None, #list of lists separating different hierarchical losses e.g. [[0.6],[6,-1]] for EC level 1 [0:6] and level 2 [6:-1]

"arch": "AWD_LSTM", # AWD_LSTM, Transformer, TransformerXL, BERT (BERT shares params nh, nl, dropout with the LSTM config)
"nheads":6, # number of BERT/Transformer heads 

"tie_encoder":True, # tie embedding and output for LMs

"max_seq_len":1024, # max. sequence length (required for certain truncation modes and BERT)
"truncation_mode":0, #determines how to deal with long sequences (longer than max_seq_len) 0: keep (default for RNN) 1:keep first max_seq_len tokens (default for BERT) 2:keep last max_seq_len tokens (not consistent with BERT) 3: drop

"eval_on_test":False, #use original test set as validation set (should only be used for train=False or together with concat_train_val)
"eval_on_val_test":False, #use validation set as usual and evaluate on the test set at the end
"concat_train_val":False, #use both train and valid as training set (for fit using all data- has to be used in conjuction with eval_on_test=True)
"metrics":["accuracy"], # array of strings specifying metrics for evaluation (currently supported accuracy, macro_auc, macro_f1, binary_auc, binary_auc50)
"export_preds":False, #outputs validation set predictions as preds.npz (bundled numpy array with entries val_IDs_sorted (validation set IDs corresponding to the entries in IDs.npy), preds (validation set predictions- these are the actual model outputs and might require an additional softmax), targs (the corresponding target labels))

"annotation":False, # for classification only: this is an annotation task (one output per token)
"regression":False, # for classification only: this is a regression task

"early_stopping": "None", #performs early stopping on specified metric (possible entries: valid entries for metrics or trn_loss or val_loss)

"interactive": False, # for execution in juyter environment; allows manual determination of lrs (specifying just the lr for the first finetuning step)
"interactive_finegrained":False, # for execution in juyter environment; allows manual determination of lrs (specifying lrs for all finetuning steps)

"return_learner": False #returns learner and exits
}



kwargs_udsmprot_customec = {
"cv_fold":-1, #cv-fold -1 for single split else fold
"working_folder":"/work3/felteu/data/", # folder with preprocessed data 
"pretrained_folder":"/work3/felteu/repos/UDSMProt/pretrained_models/lm_sprot_uniref_fwd", # folder for pretrained model
"model_filename_prefix":"model", # filename for saved model
"model_folder":"",# folder for model to evaluate if train=False (empty string refers to data folder) i.e. model will be located at {model_folder}/models/{model_filename_prefix}_3.pth
"pretrained_model_filename":"model_3_enc", # filename of pretrained model (default for loading a lm encoder); a suffix _enc will load the encoder only otherwise the full model will be loaded

"emb_sz":400, # embedding size
"nh":1150, # number of hidden units
"nl":3, # number of layers
"lin_ftrs":None, #optional list of hidden layer sizes for the classification head (None == [50])

"wd":1e-7, # weight decay
"bptt":70, # backpropagation through time (sequence length fed at a time) in fastai LMs is approximated to a std. deviation around 70, by perturbing the sequence length on a per-batch basis
"max_len":1024, # RNN only- number of tokens for which the loss is backpropagated (last max_len tokens of the sequence) [only for ordinary classification i.e. annotation=False]  see bptt for classification in the paper   BE CAREFUL: HAS TO BE LARGE ENOUGH
"bs":128, # batch size
"dropout":0.5, # dropout rate
"lr":1e-3, # learning rate
"lr_slice_exponent": 2.6, # learning rate decay per layer group (used if interactive_finegrained=False); set to 1 to disable discriminative lrs
"lr_fixed":False, #do not reduce the lr after each finetuning step
"lr_stage_factor": [2,2,5], #if lr_fixed==False reduce lr after each stage by dividing by this factor (allows to pass a list to specify factor for every stage)
"epochs":30, # epochs for (final) tuning step (pass a list with 4 elements to specify all epochs)
"fp16":False, #fp 16 training
"fit_one_cycle": True, #use one-cycle lr scheduling

"backwards": False, #invert order of all
"clip":.25, # gradient clipping
"train":True, # False for inference only

"from_scratch":True, # train from scratch or pretrain
"gradual_unfreezing":True, # unfreeze layer by layer during finetuning

"hierarchical":None, #list of lists separating different hierarchical losses e.g. [[0.6],[6,-1]] for EC level 1 [0:6] and level 2 [6:-1]

"arch": "AWD_LSTM", # AWD_LSTM, Transformer, TransformerXL, BERT (BERT shares params nh, nl, dropout with the LSTM config)
"nheads":6, # number of BERT/Transformer heads 

"tie_encoder":True, # tie embedding and output for LMs

"max_seq_len":1024, # max. sequence length (required for certain truncation modes and BERT)
"truncation_mode":0, #determines how to deal with long sequences (longer than max_seq_len) 0: keep (default for RNN) 1:keep first max_seq_len tokens (default for BERT) 2:keep last max_seq_len tokens (not consistent with BERT) 3: drop

"eval_on_test":False, #use original test set as validation set (should only be used for train=False or together with concat_train_val)
"eval_on_val_test":False, #use validation set as usual and evaluate on the test set at the end
"concat_train_val":False, #use both train and valid as training set (for fit using all data- has to be used in conjuction with eval_on_test=True)
"metrics":["accuracy"], # array of strings specifying metrics for evaluation (currently supported accuracy, macro_auc, macro_f1, binary_auc, binary_auc50)
"export_preds":False, #outputs validation set predictions as preds.npz (bundled numpy array with entries val_IDs_sorted (validation set IDs corresponding to the entries in IDs.npy), preds (validation set predictions- these are the actual model outputs and might require an additional softmax), targs (the corresponding target labels))

"annotation":False, # for classification only: this is an annotation task (one output per token)
"regression":False, # for classification only: this is a regression task

"early_stopping": "None", #performs early stopping on specified metric (possible entries: valid entries for metrics or trn_loss or val_loss)

"interactive": False, # for execution in juyter environment; allows manual determination of lrs (specifying just the lr for the first finetuning step)
"interactive_finegrained":False, # for execution in juyter environment; allows manual determination of lrs (specifying lrs for all finetuning steps)

"return_learner": False #returns learner and exits
}