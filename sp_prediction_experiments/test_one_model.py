'''
Still don't trust the new metric module- needs more checks!
'''
import torch
import os
import pandas as pd
import numpy as np
import sys
from typing import List
import logging
import argparse
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet

from models.sp_tagging_prottrans import XLNetSequenceTaggingCRF, ProteinXLNetTokenizer
from train_scripts.utils.signalp_dataset import PartitionThreeLineFastaDataset, LargeCRFPartitionDataset
from train_scripts.downstream_tasks.metrics_utils import get_metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S")
c_handler.setFormatter(formatter)
logger.addHandler(c_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cross-validate XLNet SP CRF')
    parser.add_argument('--checkpoint_path', type = str, default = '/work3/felteu/tagging_checkpoints/xlnet')
    parser.add_argument('--data', type = str, default = '/work3/felteu/data/signalp_5_data/full/train_set.fasta')
    parser.add_argument('--output_dir', type = str, default = 'crossval_results')
    parser.add_argument('--test_on_set', type = int, default = 1)
    args = parser.parse_args()


    tokenizer = ProteinXLNetTokenizer.from_pretrained('Rostlab/prot_xlnet', do_lower_case = False)
    model = XLNetSequenceTaggingCRF.from_pretrained(args.checkpoint_path)
    if model.use_large_crf:
        ds = LargeCRFPartitionDataset(args.data, tokenizer, partition_id = [args.test_on_set], add_special_tokens = True)
    else:
        ds = PartitionThreeLineFastaDataset(args.data, tokenizer, partition_id = [args.test_on_set], add_special_tokens = True)

    dataloader = torch.utils.data.DataLoader(ds, collate_fn = ds.collate_fn, batch_size = 80)
    metrics = get_metrics(model,dataloader)
    logger.info(metrics)
