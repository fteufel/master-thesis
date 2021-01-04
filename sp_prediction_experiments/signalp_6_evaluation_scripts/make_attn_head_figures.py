'''
Make plots for attention alignment
https://github.com/salesforce/provis/blob/e67812286136fe02502c6fddf311198bf43077ec/protein_attention/attention_analysis/report_edge_features.py
Attention alignments computed in own script, takes a lot of compute. Here loaded as pickle.
'''
import pickle
import seaborn as sns
import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../')
from train_scripts.utils.signalp_label_processing import SP_REGION_VOCAB

revdict = dict(zip(SP_REGION_VOCAB.values(), SP_REGION_VOCAB.keys()))
cmap = 'Blues'



def make_plots(aligned, totals, prefix, args):
    #state groupings
    groupdict ={}
    groupdict['Sec/SPI'] = [3,4,5]
    groupdict['Sec/SPII'] = [9,8,10,11,12]
    groupdict['Sec/SPIII'] = [31,32,33]
    groupdict['Tat/SPI'] = [16,17,18,19]
    groupdict['Tat/SPII'] = [23,24,25,26,27]
    groupdict['None'] = [0,1,2]

    plt.figure(figsize=(15,8))
    for idx, lab in enumerate(groupdict):

        #aggregate
        aligned_sum = np.stack([aligned[k] for k in groupdict[lab]]).sum(axis=0) 
        totals_sum = np.stack([totals[k] for k in groupdict[lab]]).sum(axis=0)
        
        
        plt.subplot(2,3,idx+1)
        heatmap = sns.heatmap(aligned_sum/totals_sum * 100, cmap=cmap,   linewidths=.02, linecolor='#D0D0D0',
                            rasterized=False, vmin=0, #cbar_kws={'label': 'Attention focused on feature [%]'},
                            #only every 2nd label
                            yticklabels=[str(x) if x%4==0 else '' for x in range(1,aligned_sum.shape[0]+1)],
                            xticklabels=[str(x) if x%2==0 else '' for x in range(1,aligned_sum.shape[1]+1)])
        plt.setp(heatmap.get_yticklabels(), fontsize=10)
        plt.setp(heatmap.get_xticklabels(), fontsize=10)
        plt.yticks(rotation=0) 

        heatmap.invert_yaxis()
        #heatmap.set_facecolor('#E7E6E6')
        plt.title(f'% Attention on {lab}')
        if idx in [0,3]:
            plt.ylabel('Layer', size=14)
        if idx in [3,4,5]:
            plt.xlabel('Head', size=14)
            
        for _, spine in heatmap.spines.items():
            spine.set_visible(True)
            
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, prefix+'attention_per_class.png'))



    labels_to_plot = ['SP_N', 'LIPO_N', 'TAT_N', 'TATLIPO_N']
    nice_label_strings = ['Sec/SPI n-region', 'Sec/SPII n-region', 'Tat/SPI n-region', 'Tat/SPII n-region']
    plt.figure(figsize=(10,8))
    for idx, lab in enumerate(labels_to_plot):

        #aggregate
        k = SP_REGION_VOCAB[lab]
        
        
        plt.subplot(2,2,idx+1)
        heatmap = sns.heatmap(aligned[k]/totals[k] * 100, cmap=cmap,   linewidths=.02, linecolor='#D0D0D0',
                            rasterized=False, vmin=0, #cbar_kws={'label': 'Attention focused on feature [%]'},
                            #only every 2nd label
                            yticklabels=[str(x) if x%4==0 else '' for x in range(1,aligned_sum.shape[0]+1)],
                            xticklabels=[str(x) if x%2==0 else '' for x in range(1,aligned_sum.shape[1]+1)])
        plt.setp(heatmap.get_yticklabels(), fontsize=10)
        plt.setp(heatmap.get_xticklabels(), fontsize=10)
        plt.yticks(rotation=0) 

        heatmap.invert_yaxis()
        #heatmap.set_facecolor('#E7E6E6')
        plt.title(f'% Attention on {nice_label_strings[idx]}')
        if idx in [0,2]:
            plt.ylabel('Layer', size=14)
        if idx in [2,3]:
            plt.xlabel('Head', size=14)
            
        for _, spine in heatmap.spines.items():
            spine.set_visible(True)
        
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, prefix+'attention_n_region.png'))


    labels_to_plot = ['SP_H', 'LIPO_H', 'TAT_H', 'TATLIPO_H']
    nice_label_strings = ['Sec/SPI h-region', 'Sec/SPII h-region', 'Tat/SPI h-region', 'Tat/SPII h-region']
    plt.figure(figsize=(10,8))
    for idx, lab in enumerate(labels_to_plot):

        #aggregate
        k = SP_REGION_VOCAB[lab]
        
        
        plt.subplot(2,2,idx+1)
        heatmap = sns.heatmap(aligned[k]/totals[k] * 100, cmap=cmap,   linewidths=.02, linecolor='#D0D0D0',
                            rasterized=False, vmin=0, #cbar_kws={'label': 'Attention focused on feature [%]'},
                            #only every 2nd label
                            yticklabels=[str(x) if x%4==0 else '' for x in range(1,aligned_sum.shape[0]+1)],
                            xticklabels=[str(x) if x%2==0 else '' for x in range(1,aligned_sum.shape[1]+1)])
        plt.setp(heatmap.get_yticklabels(), fontsize=10)
        plt.setp(heatmap.get_xticklabels(), fontsize=10)
        plt.yticks(rotation=0) 

        heatmap.invert_yaxis()
        #heatmap.set_facecolor('#E7E6E6')
        plt.title(f'% Attention on {nice_label_strings[idx]}')
        if idx in [0,2]:
            plt.ylabel('Layer', size=14)
        if idx in [2,3]:
            plt.xlabel('Head', size=14)
            
        for _, spine in heatmap.spines.items():
            spine.set_visible(True)
            
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, prefix+'attention_h_region.png'))



    labels_to_plot = ['SP_C', 'LIPO_CS', 'TAT_C', 'TATLIPO_CS']
    nice_label_strings = ['Sec/SPI c-region', 'Sec/SPII conserved c-terminus', 'Tat/SPI c-region', 'Tat/SPII conserved c-terminus']
    plt.figure(figsize=(10,8))
    for idx, lab in enumerate(labels_to_plot):

        #aggregate
        k = SP_REGION_VOCAB[lab]
        
        
        plt.subplot(2,2,idx+1)
        heatmap = sns.heatmap(aligned[k]/totals[k] * 100, cmap=cmap,   linewidths=.02, linecolor='#D0D0D0',
                            rasterized=False, vmin=0, #cbar_kws={'label': 'Attention focused on feature [%]'},
                            #only every 2nd label
                            yticklabels=[str(x) if x%4==0 else '' for x in range(1,aligned_sum.shape[0]+1)],
                            xticklabels=[str(x) if x%2==0 else '' for x in range(1,aligned_sum.shape[1]+1)])
        plt.setp(heatmap.get_yticklabels(), fontsize=10)
        plt.setp(heatmap.get_xticklabels(), fontsize=10)
        plt.yticks(rotation=0) 

        heatmap.invert_yaxis()
        #heatmap.set_facecolor('#E7E6E6')
        plt.title(f'% Attention on {nice_label_strings[idx]}')
        if idx in [0,2]:
            plt.ylabel('Layer', size=14)
        if idx in [2,3]:
            plt.xlabel('Head', size=14)
            
        for _, spine in heatmap.spines.items():
            spine.set_visible(True)
            
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, prefix+'attention_c_region.png'))



    ## grouped
    labels_to_plot = ['TAT_RR', 'TATLIPO_RR', 'LIPO_C1', 'TATLIPO_C1']
    nice_label_strings = ['Tat/SPI twin-R', 'Tat/SPII twin-R', 'Sec/SPII +1 C', 'Tat/SPII +1 C']
    plt.figure(figsize=(10,8))
    for idx, lab in enumerate(labels_to_plot):

        #aggregate
        k = SP_REGION_VOCAB[lab]
        
        
        plt.subplot(2,2,idx+1)
        heatmap = sns.heatmap(aligned[k]/totals[k] * 100, cmap=cmap,   linewidths=.02, linecolor='#D0D0D0',
                            rasterized=False, vmin=0, #cbar_kws={'label': 'Attention focused on feature [%]'},
                            #only every 2nd label
                            yticklabels=[str(x) if x%4==0 else '' for x in range(1,aligned_sum.shape[0]+1)],
                            xticklabels=[str(x) if x%2==0 else '' for x in range(1,aligned_sum.shape[1]+1)])
        plt.setp(heatmap.get_yticklabels(), fontsize=10)
        plt.setp(heatmap.get_xticklabels(), fontsize=10)
        plt.yticks(rotation=0) 

        heatmap.invert_yaxis()
        #heatmap.set_facecolor('#E7E6E6')
        plt.title(f'% Attention on {nice_label_strings[idx]}')
        if idx in [0,2]:
            plt.ylabel('Layer', size=14)
        if idx in [2,3]:
            plt.xlabel('Head', size=14)
            
        for _, spine in heatmap.spines.items():
            spine.set_visible(True)
            
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, prefix+'attention_special_regions.png'))


## grouped
    labels_to_plot = ['PILIN_P', 'PILIN_CS', 'PILIN_H', 'PILIN_O']
    nice_label_strings = ['Sec/SPIII cleaved peptide', 'Sec/SPIII conserved region', 'Sec/SPIII hydrophobic region', 'Sec/SPIII extracellular']
    plt.figure(figsize=(10,8))
    for idx, lab in enumerate(labels_to_plot):

        #aggregate
        k = SP_REGION_VOCAB[lab]
        
        
        plt.subplot(2,2,idx+1)
        heatmap = sns.heatmap(aligned[k]/totals[k] * 100, cmap=cmap,   linewidths=.02, linecolor='#D0D0D0',
                            rasterized=False, vmin=0, #cbar_kws={'label': 'Attention focused on feature [%]'},
                            #only every 2nd label
                            yticklabels=[str(x) if x%4==0 else '' for x in range(1,aligned_sum.shape[0]+1)],
                            xticklabels=[str(x) if x%2==0 else '' for x in range(1,aligned_sum.shape[1]+1)])
        plt.setp(heatmap.get_yticklabels(), fontsize=10)
        plt.setp(heatmap.get_xticklabels(), fontsize=10)
        plt.yticks(rotation=0) 

        heatmap.invert_yaxis()
        #heatmap.set_facecolor('#E7E6E6')
        plt.title(f'% Attention on {nice_label_strings[idx]}')
        if idx in [0,2]:
            plt.ylabel('Layer', size=14)
        if idx in [2,3]:
            plt.xlabel('Head', size=14)
            
        for _, spine in heatmap.spines.items():
            spine.set_visible(True)
            
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, prefix+'attention_spiii_regions.png'))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--attention_alignment_pickle', type=str, default = 'experiments_results/bertology/t0v1_attention_alignment_results.pkl')
    parser.add_argument('--attention_alignment_pickle_untrained', type=str, default = 'experiments_results/bertology/prottrans_bert_attention_alignment_results.pkl')


    parser.add_argument('--output_dir', type=str, default = 'attention_alignment_plots')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f'made {args.output_dir}')
        os.makedirs(args.output_dir)


    aligned, totals =  pickle.load(open(args.attention_alignment_pickle, 'rb'))
    make_plots(aligned,totals,'t0v1_', args)


    aligned, totals =  pickle.load(open(args.attention_alignment_pickle_untrained, 'rb'))
    make_plots(aligned,totals,'prot_bert_', args)


if __name__ == '__main__':
    main()