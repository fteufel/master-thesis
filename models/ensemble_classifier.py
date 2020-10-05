'''
Utility to get predictions from multiple models and average the predictions.
Implemented as nn.Module so that we can use model.to(device) on the ensemble classifier.

TODO this is convenient, but blows up gpu memory. Can't use.
'''

import torch
import torch.nn as nn
import numpy as np
from typing import List
import os

class EnsembleModel(nn.Module):
    def __init__(self, models:List[nn.Module]):
        super().__init__()

        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        '''Predicts with each model in models and averages outputs'''
        #get predictions from all models
        result_list = []
        for model in self.models:
            results = model(*args, **kwargs)
            result_list.append(results)

        #average the predictions
        output_obj = list(zip(*result_list)) #repacked

        avg_list = []
        for obj in output_obj:
            #identify type - does not need to be tensor
            if type(obj[0]) == torch.Tensor:
                avg = torch.stack(obj).float().mean(axis=0) #call float to avoid error when dealing with long tensors
            elif type(obj[0]) == np.ndarray:
                avg = np.stack(obj).mean(axis=0)
            else:
                raise NotImplementedError

            avg_list.append(avg)

        return avg_list


def make_ensemble_model(model: nn.Module, base_path, model_names: List[str]) -> nn.Module:
    '''Load multiple models and create an ensemble model that averages predictions.
       Works for huggingface-style models.
    '''
    model_list = []
    for name in model_names:
        model_instance =model.from_pretrained(os.path.join(base_path, name))
        model_list.append(model_instance)

    return EnsembleModel(model_list)

#outer best_part_1  best_part_2  best_part_4
#inner test_1_val_0  test_1_val_2  test_1_val_3  test_1_val_4

def make_ensemble_model_crossval(model: nn.Module, base_path) -> nn.Module:
    partitions = [0,1,2,3,4]
    model_list = []

    for outer_part in partitions:
        for inner_part in partitions:

            if inner_part != outer_part:
                path = os.path.join(base_path, f'best_part_{outer_part}', f'test_{outer_part}_val_{inner_part}')
                model_instance = model.from_pretrained(path)
                model_list.append(model_instance)

    assert len(model_list) == 20
    return EnsembleModel(model_list)