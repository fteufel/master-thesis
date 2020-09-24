from .mixout_module_xlnet import MixXLNetRelativeAttention, MixLinear
from transformers.modeling_xlnet import XLNetRelativeAttention #just for type checking
import torch.nn as nn
from IPython import embed

def recursive_set_module(model, name, new_module):
    '''Helper function to deal with a nested model structure'''
    name_splitted = name.split('.')
    current_level_module = model
    for i, lvl_name  in enumerate(name_splitted):
        #get module of level
        current_level_module = getattr(current_level_module, lvl_name) if type(lvl_name) == str else model[lvl_name] #when name is a int, must be list

        #when reaching the last level, set.
        if i == len(name_splitted) -2:
            setattr(current_level_module, name_splitted[-1], new_module)
            break

def apply_mixout_to_xlnet(model, mixout_prob):

    for name, module in list(model.named_modules()):

        name_split = name.split('.') # [transformer, layer, {number}, {module_name}]

        if name_split[-1] == 'dropout' and isinstance(module, nn.Dropout): #this should get all pretrained transformer dropouts
            recursive_set_module(model, name, nn.Dropout(0))

        #change linear layers
        elif 'layer_' in name_split[-1] and isinstance(module, nn.Linear):
            target_state_dict = module.state_dict()
            bias = True if module.bias is not None else False
            new_module = MixLinear(module.in_features, module.out_features, 
                                bias, target_state_dict['weight'], mixout_prob)
            new_module.load_state_dict(target_state_dict)
            #setting by name does not work - module has a list, repr name is different from indexing
            recursive_set_module(model, name, new_module)

        #change transformer layers
        elif name_split[-1] == 'rel_attn' and isinstance(module, XLNetRelativeAttention):
            target_state_dict = module.state_dict()

            new_module = MixXLNetRelativeAttention(config=model.config, targets = target_state_dict, mixout_prob = mixout_prob)
            recursive_set_module(model, name, new_module)

        else:
            print(f'Not changing {name}: {type(module)}')


        

