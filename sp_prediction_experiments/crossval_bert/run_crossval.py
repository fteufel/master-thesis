import pandas as pd
import torch
import argparse
import sys
import os
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet

from train_scripts.utils.signalp_dataset import LargeCRFPartitionDataset
from models.sp_tagging_prottrans import BertSequenceTaggingCRF, ProteinBertTokenizer
from train_scripts.downstream_tasks.metrics_utils import get_metrics






if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get ensemble predictions')
    parser.add_argument('--data_path', type=str, default='/work3/felteu/data/signalp_5_data/full/train_set.fasta')
    parser.add_argument('--base_path', type=str, default='/work3/felteu/tagging_checkpoints/bert_crossval')
    parser.add_argument('--output_path', type=str,default='/work3/felteu/preds')
    args=parser.parse_args()

    tokenizer = ProteinBertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)

    result_list = []
    part_names = []
    partitions = [0,1,2,3,4]

    for outer_part in partitions:
        ds = LargeCRFPartitionDataset(args.data_path,tokenizer=tokenizer,partition_id=outer_part,add_special_tokens=True,return_kingdom_ids=True)
        dataloader = torch.utils.data.DataLoader(ds, collate_fn = ds.collate_fn, batch_size = 100)
        for inner_part in partitions:
            if inner_part != outer_part:
                path = os.path.join(args.base_path, f'best_part_{outer_part}', f'test_{outer_part}_val_{inner_part}')
                part_names.append(f'T{outer_part}V{inner_part}')
                model_instance = BertSequenceTaggingCRF.from_pretrained(path)
                metrics = get_metrics(model_instance,dataloader)
                result_list.append(metrics)

    df = pd.DataFrame.from_dict(result_list)
    df.index = part_names
    df.T.to_csv(os.path.join(args.output_path,'crossval_results.csv')
