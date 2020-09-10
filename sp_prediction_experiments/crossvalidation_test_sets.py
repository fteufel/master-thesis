import torch
import os
import pandas as pd
import sys
from typing import List
import logging
import argparse
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") #to make it work on hpc, don't want to install in venv yet

from models.sp_tagging_prottrans import XLNetSequenceTaggingCRF, XLNetTokenizer, ProteinXLNetTokenizer
from train_scripts.utils.signalp_dataset import PartitionThreeLineFastaDataset
from train_scripts.downstream_tasks.sp_tagging_metrics_utils import validate

#annotation [S: Sec/SPI signal peptide | T: Tat/SPI signal peptide | L: Sec/SPII signal peptide | I: cytoplasm | M: transmembrane | O: extracellular]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S")
c_handler.setFormatter(formatter)
logger.addHandler(c_handler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cross_validate_xlnet(checkpoint_dir: str, data_path: str,
                         partition_id: List[str] = [0,1,2,3,4],
                         kingdom_id: List[str] = ['EUKARYA', 'ARCHAEA', 'NEGATIVE', 'POSITIVE'],
                         type_id: List[str] = ['LIPO', 'NO_SP', 'SP', 'TAT'], ) -> pd.core.frame.DataFrame:
    '''Load checkpoints, take mean of inner splits,
    do for all outer splits and and aggregate mean and standard deviation.
    Inputs:
        dataloader: torch dataloader to test on
        checkpoint_dir: base directory where checkpoints of format "crossval_test_0_valid_1" are saved
    Output:
        pd.dataframe of format
        idx | metric a | metric b | ...
        ----|----------|----------|----
          0 |  0.87    |  0.664   |

    '''
    checkpoint_type = 'best_mcc_sum' #subdirectory, choose which saved checkpoint to load
    tokenizer = ProteinXLNetTokenizer.from_pretrained('Rostlab/prot_xlnet', do_lower_case = False)

    partitions_list = [0,1,2,3,4]
    outer_results = [] # list of pd series
    for i in partitions_list: # test partition
        ds = PartitionThreeLineFastaDataset(data_path, tokenizer, partition_id = [i],kingdom_id=kingdom_id,type_id = type_id, add_special_tokens = True)
        dataloader = torch.utils.data.DataLoader(ds, collate_fn = ds.collate_fn, batch_size = 40)

        inner_results = [] #list of dicts
        for j in partitions_list: #validation partition

            if not i == j:
                checkpoint_path =  os.path.join(checkpoint_dir, f'crossval_test_{i}_valid_{j}', checkpoint_type)
                model = XLNetSequenceTaggingCRF.from_pretrained(checkpoint_path)
                model.to(device)
                loss, metrics = validate(model, dataloader)
                logger.info(f'Testing on {i}, holdout {j}')
                logger.info(metrics)
                inner_results.append(metrics)

                #list of dicts, process.
                df = pd.DataFrame.from_dict(inner_results)
                mean_series = df.mean()
                outer_results.append(mean_series)

    df = pd.DataFrame(outer_results)

    return df



def main(args):
    mean_list = []  #list of pd.series
    stds_list = []   #list of pd.series
    index_list = [] #list of str, save name for each result
    #EUKARYA SPI
    logger.info(f'Processing EUKARYA SPI')
    df = cross_validate_xlnet(args.checkpoint_dir, args.data, kingdom_id = ['EUKARYA'], type_id = ['SP', 'NO_SP'])
    means, stds = df.mean() , df.std()
    mean_list.append(means)
    stds_list.append(stds)
    index_list.append('eukarya SPI')

    for kingdom in ['ARCHAEA', 'POSITIVE', 'NEGATIVE']:
        #SPI
        logger.info(f'Processing {kingdom} SPI')
        df = cross_validate_xlnet(args.checkpoint_dir, args.data, kingdom_id = [kingdom], type_id = ['SP', 'NO_SP'])
        means, stds = df.mean() , df.std()
        mean_list.append(means)
        stds_list.append(stds)
        index_list.append(f'{kingdom.lower()} SPI')
        #SPII
        logger.info(f'Processing {kingdom} SPII')
        df = cross_validate_xlnet(args.checkpoint_dir, args.data, kingdom_id = [kingdom], type_id = ['LIPO', 'NO_SP'])
        means, stds = df.mean() , df.std()
        mean_list.append(means)
        stds_list.append(stds)
        index_list.append(f'{kingdom.lower()} SPII')
        #TAT
        logger.info(f'Processing {kingdom} TAT')
        df = cross_validate_xlnet(args.checkpoint_dir, args.data, kingdom_id = [kingdom], type_id = ['TAT', 'NO_SP'])
        means, stds = df.mean() , df.std()
        mean_list.append(means)
        stds_list.append(stds)
        index_list.append(f'{kingdom.lower()} TAT')

    mean_df = pd.DataFrame(mean_list, index = index_list)
    stds_df = pd.DataFrame(stds_list, index = index_list)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    mean_df.to_csv(os.path.join(args.output_dir, 'crossval_means.csv'))
    stds_df.to_csv(os.path.join(args.output_dir, 'crossval_stdevs.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cross-validate XLNet SP CRF')
    parser.add_argument('--checkpoint_dir', type = str, default = '/work3/felteu/tagging_checkpoints/xlnet')
    parser.add_argument('--data', type = str, default = '/work3/felteu/data/signalp_5_data/full/train_set.fasta')
    parser.add_argument('--output_dir', type = str, default = 'crossval_results')
    args = parser.parse_args()

    

    main(args)