import torch
import torch.nn as nn
import numpy as np
from typing import List
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_data(model, dataloader):
    '''run all the data of a DataLoader, concatenate and return outputs
    Extended from metrics_utils, need more variables here'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_targets = []
    all_global_targets = []
    all_global_probs = []
    all_pos_preds = []
    all_losses = []

    total_loss = 0
    for i, batch in enumerate(dataloader):

        if hasattr(dataloader.dataset, 'kingdom_ids') and hasattr(dataloader.dataset, 'sample_weights'):
            data, targets, input_mask, global_targets, sample_weights, kingdom_ids = batch
            kingdom_ids = kingdom_ids.to(device)
            sample_weights = sample_weights.to(device)
        elif  hasattr(dataloader.dataset, 'sample_weights') and not hasattr(dataloader.dataset, 'kingdom_ids'):
            data, targets, input_mask, global_targets, sample_weights = batch
            kingdom_ids = None
        else:
            data, targets, input_mask, global_targets, sample_weights = batch
            kingdom_ids = None
            
        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)
        global_targets = global_targets.to(device)
        with torch.no_grad():
            loss, global_loss, global_probs, pos_probs, pos_preds = model(data, 
                                                                            global_targets = global_targets, 
                                                                            targets=  targets, 
                                                                            input_mask = input_mask, 
                                                                            kingdom_ids=kingdom_ids,
                                                                            return_both_losses=True,
                                                                            return_element_wise_loss=True
                                                                         )


        #average the loss over sequence length
        input_mask = input_mask[:,1:-1]
        summed_loss = (loss *input_mask).sum(dim =1) #sum probs for each label over axis
        sequence_lengths = input_mask.sum(dim =1)
        mean_loss = summed_loss/sequence_lengths

        all_losses.append(mean_loss.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        all_global_targets.append(global_targets.detach().cpu().numpy())
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())


    all_losses = np.concatenate(all_losses)
    all_targets = np.concatenate(all_targets)
    all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)

    return all_losses, all_global_targets, all_global_probs, all_targets, all_pos_preds

def run_data_array(model, sequence_data_array, batch_size = 10):
    '''run all the data of a np.array, concatenate and return outputs
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_global_probs = []
    all_pos_preds = []

    #SIGNALP_KINGDOM_DICT = {'EUKARYA': 0, 'POSITIVE':1, 'NEGATIVE':2, 'ARCHAEA':3}

    total_loss = 0
    b_start = 0
    b_end = batch_size
    while b_end < len(sequence_data_array):

        data = sequence_data_array[b_start:b_end,:]
        data = torch.tensor(data)
        data = data.to(device)
        input_mask = None
        #eukarya-sp
        kingdom_ids = torch.zeros(data.shape[0], dtype=int).to(device)
        global_targets= torch.ones(data.shape[0], dtype=int).to(device)


        with torch.no_grad():
            global_probs, pos_probs, pos_preds = model(data, global_targets = None, input_mask = input_mask,
                                                            kingdom_ids = kingdom_ids)

        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())

        b_start = b_start + batch_size
        b_end = b_end + batch_size


    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)

    return all_global_probs, all_pos_preds



from tqdm import tqdm
def run_data_ensemble(model: nn.Module, base_path, dataloader: torch.utils.data.DataLoader, data_array=None):
    result_list = []

    partitions = [0,1,2,3,4]

    checkpoint_list = []
    for outer_part in partitions:
        for inner_part in partitions:
            if inner_part != outer_part:
                path = os.path.join(base_path, f'best_part_{outer_part}', f'test_{outer_part}_val_{inner_part}')
                checkpoint_list.append(path)


    for path in tqdm(checkpoint_list):
        model_instance = model.from_pretrained(path)
        if data_array is not None:
            results = run_data_array(model_instance, data_array)
        else:
            results = run_data(model_instance, dataloader)
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

        
