'''
Script to make the figures to be used in the manuscript.
Paths to files containing raw data are hardcoded.
'''
import h5py
import sys
sys.path.append('../../../')
from train_scripts.utils.signalp_dataset import RegionCRFDataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


###
# Figure 1
###

## Make the t-SNE projection of pretrained hidden states.
ds = RegionCRFDataset('../../../data/signal_peptides/signalp_updated_data/signalp_6_train_set.fasta', partition_id = [0])
labels = ds.global_labels


h5f = h5py.File('../bertology/prottrans_bert_hidden_states.hdf5', 'r')
h5f['hidden_states']
#raw averaged, no removal of special tokens
embedded_pre = TSNE().fit_transform(h5f['hidden_states'][()].mean(axis=1))

plt.figure(figsize=(3,3))
ax = plt.gca()
sns.scatterplot(x = embedded_pre[:,0], y= embedded_pre[:,1], hue=labels, s=10, hue_order =np.array(['NO_SP','SP','LIPO',  'TAT', 'TATLIPO','PILIN',]))#, sizes=[20, 20, 20, 100, 20, 40], size=labels)
#plt.title('Pretrained Bert')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
ax.yaxis.set_label_position("right")

handles, labs = plt.gca().get_legend_handles_labels()
plt.legend(handles, np.array(['None', 'Sec/SPI', 'Sec/SPII', 'Tat/SPI', 'Tat/SPII', 'Sec/SPIII']), 
            loc='center',bbox_to_anchor=(0.5, -0.2,0,0), ncol=3, columnspacing=0.4)

plt.savefig('hidden_state_tsne.png', bbox_inches='tight')

###
# Figure 2
###

## Make the MCC barplot
metrics_1 =  pd.read_csv('../signalp_5_model/crossval_metrics.csv', index_col=0).mean(axis=1)
df =  pd.read_csv('../signalp_6_model/crossval_metrics.csv', index_col=0)
metrics_2 = df.loc[df.index.str.contains('mcc|window')].astype(float).mean(axis=1)
df = pd.DataFrame({'SignalP 5.0':metrics_1, 'SignalP 6.0':metrics_2})

#build additional identifer columns from split index
exp_df = df.reset_index()['index'].str.split('_', expand=True)
exp_df.columns = ['kingdom', 'type', 'metric', 'no', 'window']
exp_df.index =  df.index

#put together
df = df.join(exp_df)


nice_label_dict = {'NO_SP':'Other', 'SP':'Sec/SPI', 'LIPO':'Sec/SPII','TAT':'Tat/SPI', 'TATLIPO':'Tat/SPII',  'PILIN':'Sec/SPIII', None:None}

df['type'] = df['type'].apply(lambda x: nice_label_dict[x])

plt.figure(figsize = (10.2,3.4))
ax = plt.gca()
df_plot = df
df_plot = df_plot.loc[df['metric'] == 'mcc2'][['kingdom', 'type','SignalP 5.0', 'SignalP 6.0']]
df_plot = df_plot.set_index(df_plot['kingdom'].str.slice(0,3) + '\n' + df_plot['type']) #only take first 3 letters of kingdom
df_plot = df_plot.sort_index()

#rename for legend
#df_plot = df_plot.rename({'SignalP 5.0': 'MCC SignalP 5.0', 'SignalP 6.0': 'MCC SignalP 6.0'}, axis =1)

df_plot.plot(kind='bar', ax=ax, ylim =(0,1), rot=0, color=['grey', 'orange']).legend(loc='lower left')
#plt.title('MCC 2')
plt.ylabel('MCC')
plt.tight_layout()
plt.savefig('mcc.png')



## Make the CS barplot

window_0_df = df.loc[df['window']=='0']
#https://stackoverflow.com/questions/27694221/using-python-libraries-to-plot-two-horizontal-bar-charts-sharing-same-y-axis
#https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
x1_sp6 = window_0_df.loc[window_0_df['metric']=='precision']['SignalP 6.0']
x1_sp5 = window_0_df.loc[window_0_df['metric']=='precision']['SignalP 5.0']
x2_sp6 = window_0_df.loc[window_0_df['metric']=='recall']['SignalP 5.0']
x2_sp5 = window_0_df.loc[window_0_df['metric']=='recall']['SignalP 6.0']
y = np.arange(x2_sp5.size)
yticklabels = window_0_df.loc[window_0_df['metric']=='recall']['kingdom'].str.lower().str.capitalize() +' '+ window_0_df.loc[window_0_df['metric']=='recall']['type']
width=0.35

fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(6,6))


axes[0].barh(y-width/2, x1_sp5, align='center', height=width, color='gray')
axes[0].barh(y+width/2, x1_sp6, align='center', height=width, color='orange')
axes[1].barh(y-width/2, x2_sp5, align='center', height=width, color='gray', label='SignalP 5.0')
axes[1].barh(y+width/2, x1_sp6, align='center', height=width, color='orange', label ='SignalP 6.0')
axes[0].invert_xaxis()

axes[0].set(yticks=y, yticklabels=yticklabels)
axes[1].yaxis.set_ticks_position('none') 
#axes[0].yaxis.tick_right()

# need to get rid of 0 tick for one plot, would collide
x_ticks = axes[0].xaxis.get_major_ticks()
x_ticks[0].label1.set_visible(False) ## set first x tick label invisible


axes[0].set_title('Precision')
axes[1].set_title('Recall')
axes[0].spines['right'].set_visible(False)
fig.subplots_adjust(wspace=0.0)
plt.legend(loc='lower left',bbox_to_anchor=(-0.3, -0.01))
plt.savefig('cs_window_0.png',  bbox_inches = "tight")


## Make the stratified performance plot

df = pd.read_csv('../removed_samples/pooled_preds_bert.csv', index_col=0)
bert = df['mcc']

df = pd.read_csv('../removed_samples/pooled_preds_sp5.csv', index_col=0)
sp5 = df['mcc']
counts = df['count']

df = pd.DataFrame([sp5, bert],index=['SignalP 5.0', 'SignalP 6.0']).T

df.index = [f'{x} %' for x in range(0,100,10)]
#df.index =  [x+ '\n' + f'({y})' for x,y in zip(df.index, counts) ]

plt.figure(figsize=(6,2))
ax = plt.gca()

df = df.iloc[2:]
df.plot(kind='line', ax=ax, color=['gray', 'orange'])
plt.legend(loc='lower right')
plt.ylabel('MCC (multiclass)')
plt.xlabel('Maximum identity to training set')
plt.tight_layout()
plt.savefig('pooled_multiclass_mcc.png')


###
# Figure 3
###

# Load the pre-computed region characteristics saved by make_figures.py
df = pd.read_csv('../signalp_6_model/plots/region_characteristics.csv')
plt.figure(figsize = (12 ,4))
ax = plt.subplot(1,3,1)
ax.text(-0.1, 1.05, 'A',transform=ax.transAxes, size=16, weight='bold')
plot_df = df.loc[df['Type'].isin(['SP', 'TAT'])].copy()
plot_df['Type'].replace({'TAT':'Tat/SPI', 'SP': 'Sec/SPI'}, inplace=True)
sns.boxplot(data=plot_df, y='len_n', x='Kingdom', hue='Type')

handles, labs = ax.get_legend_handles_labels()
plt.legend(handles, labs, loc='upper center', ncol=2)
plt.ylabel('Length')
plt.title('n-region')


ax = plt.subplot(1,3,2)
ax.text(-0.1, 1.05, 'B',transform=ax.transAxes, size=16, weight='bold')

plot_df = df.loc[df['Type'].isin(['SP', 'TAT'])].copy()
plot_df['Type'].replace({'TAT':'Tat/SPI', 'SP': 'Sec/SPI'}, inplace=True)
sns.boxplot(data=plot_df, y='hydrophobicity_h', x='Kingdom', hue='Type')

handles, labs = ax.get_legend_handles_labels()
plt.legend(handles, labs, loc='lower center', ncol=2)
plt.ylabel('Hydrophobicity')
plt.title('h-region')

ax = plt.subplot(1,3,3)
ax.text(-0.1, 1.05, 'C',transform=ax.transAxes, size=16, weight='bold')

plot_df = df.loc[df['Type'].isin(['SP', 'TAT'])].copy()
plot_df['Type'].replace({'TAT':'Tat/SPI', 'SP': 'Sec/SPI'}, inplace=True)
sns.boxplot(data=plot_df, y='charge_c', x='Kingdom', hue='Type')

handles, labs = ax.get_legend_handles_labels()
plt.legend(handles, labs, loc='lower center', ncol=2)
plt.ylabel('Net charge')
plt.title('c-region')

plt.tight_layout()
plt.savefig('region_boxplots.png')



#
# Make figure 2 completely in matplotlib
#

from matplotlib import gridspec

fig = plt.figure(figsize=(16,5))
gs = fig.add_gridspec(nrows=5, ncols=6, wspace=2, hspace=3.4)

ax1 = fig.add_subplot(gs[:3, :4])
ax2 = fig.add_subplot(gs[3:, :3])

# Make a sub gridspec for the CS barplot
sub_gs = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs[:,4:], hspace=0.1, wspace=0.0)
                                          #height_ratios=[1,3], width_ratios=[3,1])
ax3 = fig.add_subplot(sub_gs[0,0])
ax4 = fig.add_subplot(sub_gs[0,1], sharey=ax3)



## Stratification on ax2
df = pd.read_csv('../removed_samples/pooled_preds_bert.csv', index_col=0)
bert = df['mcc']

df = pd.read_csv('../removed_samples/pooled_preds_sp5.csv', index_col=0)
sp5 = df['mcc']
counts = df['count']

df = pd.DataFrame([sp5, bert],index=['SignalP 5.0', 'SignalP 6.0']).T

df.index = [f'{x} %' for x in range(0,100,10)]

df = df.iloc[2:]
df.plot(kind='line', ax=ax2, color=['gray', 'orange'])
ax2.set_ylabel('MCC (multiclass)')
ax2.set_xlabel('Maximum identity to training set')

ax2.legend(loc='lower right',bbox_to_anchor=(1.4, 0.1), fontsize=12)

ax2.set_ylim(0.9,1)
ax2.text(-1.3, 1.01, 'C', size=16, weight='bold')



## MCC on ax1
metrics_1 =  pd.read_csv('../signalp_5_model/crossval_metrics.csv', index_col=0).mean(axis=1)
df =  pd.read_csv('../signalp_6_model/crossval_metrics.csv', index_col=0)
metrics_2 = df.loc[df.index.str.contains('mcc|window')].astype(float).mean(axis=1)
df = pd.DataFrame({'SignalP 5.0':metrics_1, 'SignalP 6.0':metrics_2})

#build additional identifer columns from split index
exp_df = df.reset_index()['index'].str.split('_', expand=True)
exp_df.columns = ['kingdom', 'type', 'metric', 'no', 'window']
exp_df.index =  df.index

#put together
df = df.join(exp_df)


nice_label_dict = {'NO_SP':'Other', 'SP':'Sec/SPI', 'LIPO':'Sec/SPII','TAT':'Tat/SPI', 'TATLIPO':'Tat/SPII',  'PILIN':'Sec/SPIII', None:None}
df['type'] = df['type'].apply(lambda x: nice_label_dict[x])

df_plot = df
df_plot = df_plot.loc[df['metric'] == 'mcc2'][['kingdom', 'type','SignalP 5.0', 'SignalP 6.0']]
df_plot = df_plot.set_index(df_plot['kingdom'].str.slice(0,3) + '\n' + df_plot['type']) #only take first 3 letters of kingdom
df_plot = df_plot.sort_index()


df_plot.plot(kind='bar', ax=ax1, ylim =(0,1), rot=30, color=['grey', 'orange'], legend=False)

ax1.set_ylabel('MCC (one-vs-all)')
ax1.text(-1.85, 1.03, 'A', size=16, weight='bold')




## CS Precision on ax 3

window_0_df = df.loc[df['window']=='0']
window_0_df = window_0_df.sort_values(['kingdom', 'type'], ascending=False) # reverse order for barh

#https://stackoverflow.com/questions/27694221/using-python-libraries-to-plot-two-horizontal-bar-charts-sharing-same-y-axis
#https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
x1_sp6 = window_0_df.loc[window_0_df['metric']=='precision']['SignalP 6.0']
x1_sp5 = window_0_df.loc[window_0_df['metric']=='precision']['SignalP 5.0']
x2_sp6 = window_0_df.loc[window_0_df['metric']=='recall']['SignalP 5.0']
x2_sp5 = window_0_df.loc[window_0_df['metric']=='recall']['SignalP 6.0']
y = np.arange(x2_sp5.size)
yticklabels = window_0_df.loc[window_0_df['metric']=='recall']['kingdom'].str.lower().str.capitalize() +' '+ window_0_df.loc[window_0_df['metric']=='recall']['type']
width=0.35


ax3.barh(y-width/2, x1_sp5, align='center', height=width, color='gray')
ax3.barh(y+width/2, x1_sp6, align='center', height=width, color='orange')
ax4.barh(y-width/2, x2_sp5, align='center', height=width, color='gray', label='SignalP 5.0')
ax4.barh(y+width/2, x1_sp6, align='center', height=width, color='orange', label ='SignalP 6.0')
ax3.invert_xaxis()

ax3.set(yticks=y, yticklabels=yticklabels)
ax4.yaxis.set_ticks_position('none') 



# need to get rid of 0 tick for one plot, would collide
x_ticks = ax3.xaxis.get_major_ticks()
x_ticks[0].label1.set_visible(False) ## set first x tick label invisible


ax3.set_title('Precision')
ax4.set_title('Recall')
ax3.spines['right'].set_visible(False)
ax3.text(0.1, 1.3, 'B', size=16, transform=ax.transAxes,weight='bold')

plt.show()
plt.savefig('figure_2_complete.png',bbox_inches = "tight")