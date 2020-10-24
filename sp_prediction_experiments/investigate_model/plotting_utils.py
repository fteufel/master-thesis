'''
Utility functions for model analysis + plotting

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def fix_eukarya_preds(df):
    '''Sum up all SP class probabilities and make them the probability for SPI'''
    
    partitions = [0,1,2,3,4]
    
    inner_df = df.loc[df['Kingdom']== 'EUKARYA']
    for p in partitions:
        for v in partitions:
            if p != v:
                
                summed_probs = inner_df[f'p_SPI model T{p}V{v}'] + inner_df[f'p_SPII model T{p}V{v}']  + inner_df[f'p_TAT model T{p}V{v}']
                #inner_df[f'p_SPII model T{p}V{v}']  =0.
                #inner_df[f'p_TAT model T{p}V{v}'] = 0.
    
                df.loc[np.where(df['Kingdom']== 'EUKARYA')[0],f'p_SPI model T{p}V{v}'] = summed_probs
                df.loc[np.where(df['Kingdom']== 'EUKARYA')[0],f'p_SPII model T{p}V{v}'] = 0.
                df.loc[np.where(df['Kingdom']== 'EUKARYA')[0],f'p_TAT model T{p}V{v}'] = 0.

    
    return df



def make_conf_matrix(y_true, y_pred, percent=False, categories= ['Other', 'Sec\nSPI', 'Sec\nSPII' ,'Tat\nSPI'], label_size = 10, axlabel_size=10, tick_size=10, ax=None):
    confusion = confusion_matrix(y_true, y_pred)
    if percent:
        confusion_norm = confusion/confusion.sum(axis=1)[:, None]
        
        group_counts = ['{0:0.0f}'.format(value) for value in confusion.flatten()]
        group_percentages = ['{0:.1%}'.format(value) for value in confusion_norm.flatten()]
        labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(len(categories),len(categories))
        
        b= sns.heatmap(confusion_norm, cmap="Blues", annot=labels, xticklabels=categories,yticklabels=categories, fmt='', cbar=False,
                   annot_kws={"size": label_size}, ax=ax)

    else:
        b= sns.heatmap(confusion, cmap="Blues", annot= True, xticklabels=categories,yticklabels=categories)
    
    #resize labels
    #b.set_yticklabels(b.get_yticks(), size = tick_size)
    #b.set_xticklabels(b.get_xticks(), size= tick_size)

    #plt.ylabel('True label', size=axlabel_size)
    #plt.xlabel('Predicted label', size=axlabel_size)
    
    return b

def plot_conf_matrix(confusion, categories = ['Other', 'Sec\nSPI', 'Sec\nSPII' ,'Tat\nSPI'], label_size=10, axlabel_size=10, tick_size=10, ax=None):
    '''For precomputed confusion matrices'''
    confusion_norm = confusion/confusion.sum(axis=1)[:, None]
    group_counts = ['{0:0.0f}'.format(value) for value in confusion.flatten()]
    group_percentages = ['{0:.1%}'.format(value) for value in confusion_norm.flatten()]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(len(categories),len(categories))
    b= sns.heatmap(confusion_norm, cmap="Blues", annot=labels, xticklabels=categories,yticklabels=categories, fmt='', cbar=False,
                annot_kws={"size": label_size}, ax=ax)
    
    #b.set_yticklabels(b.get_yticks(), size = tick_size)
    #b.set_xticklabels(b.get_xticks(), size= tick_size)
    plt.ylabel('True label', size=axlabel_size)
    plt.xlabel('Predicted label', size=axlabel_size)
    
    return b