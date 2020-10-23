'''
Script to compute crossvalidation metrics from output .csv file produced by get_preds.py, using the --full_output flag.

Produces full 20 model csv for each input file and 1 csv with mean performances per file

'''


import pandas as pd
import sys
import argparse
import os
sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/") 
from train_scripts.downstream_tasks.metrics_utils import compute_metrics

def crossvalidatation_metrics_from_df(df):
    results_list = []
    for test_partition in [0,1,2,3,4]:
        for val_partition in [0,1,2,3,4]:
            if test_partition != val_partition:
                test_set_sequences = df.loc[df['Partition']==test_partition]#.filter(regex=f'T{test_partition}V{val_partition}', axis=1)

                all_kingdom_ids = test_set_sequences['Kingdom']
                all_global_targets = test_set_sequences['target']
                all_cs_targets = test_set_sequences['CS target']
                all_cs_preds = test_set_sequences[f'CS model T{test_partition}V{val_partition}']
                test_set_sequences.filter(regex=f'T{test_partition}V{val_partition}', axis=1)
                probs = test_set_sequences[[f'p_NO model T{test_partition}V{val_partition}', 
                                            f'p_SPI model T{test_partition}V{val_partition}', 
                                            f'p_SPII model T{test_partition}V{val_partition}', 
                                            f'p_TAT model T{test_partition}V{val_partition}']]

                probs = probs.values
                all_global_preds = probs.argmax(axis=1)

                #compute_metrics
                metrics = compute_metrics(all_global_targets, all_global_preds, all_cs_targets, all_cs_preds, all_kingdom_ids)
                results_list.append(metrics)

                crossval_df = pd.DataFrame.from_dict(results_list).T
            
    return crossval_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir')
    parser.add_argument('--files', nargs='+', help='.csv files to process', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    means = {}
    for f in args.files:
        fname = f.split('/')[-1].rstrip('.csv')
        df = pd.read_csv(f)
        crossval_df = crossvalidatation_metrics_from_df(df)
        crossval_df = crossval_df.drop(crossval_df.filter(regex='NO_SP', axis=0).index)
        crossval_df.to_csv(os.path.join(args.output_dir, f'crossval_metrics_{fname}.csv'))

        means[fname] = crossval_df.mean(axis=1)


    pd.DataFrame.from_dict(means).to_csv(os.path.join(args.output_dir, 'crossval_means.csv'))

