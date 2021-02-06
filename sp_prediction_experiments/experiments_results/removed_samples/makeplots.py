import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('remainder_stratified_crossval_bert.csv', index_col=0)
bert = df.loc[df.index.str.contains('multiclass')].mean(axis=1)

df = pd.read_csv('remainder_stratified_crossval_sp5.csv', index_col=0)
sp5 = df.loc[df.index.str.contains('multiclass')].mean(axis=1)



df = pd.DataFrame([sp5, bert],index=['SignalP 5.0', 'Bert-CRF']).T


df.index = [f'{x} %' for x in range(0,100,10)]

df = df.iloc[2:]
df.plot(kind='line', figsize=(6,4))
plt.legend(loc='lower left')
plt.ylabel('MCC')
plt.xlabel('Maximum identity to training set')
plt.tight_layout()
plt.savefig('multiclass_mcc.png')


df = pd.read_csv('pooled_preds_bert.csv', index_col=0)
bert = df['mcc']

df = pd.read_csv('pooled_preds_sp5.csv', index_col=0)
sp5 = df['mcc']
counts = df['count']

df = pd.DataFrame([sp5, bert],index=['SignalP 5.0', 'SignalP 6.0']).T

df.index = [f'{x} %' for x in range(0,100,10)]
df.index =  [x+ '\n' + f'({y})' for x,y in zip(df.index, counts) ]

df = df.iloc[2:]
df.plot(kind='line', figsize=(6,4))
plt.legend(loc='lower right')
plt.ylabel('MCC')
plt.xlabel('Maximum identity to training set')
plt.tight_layout()
plt.savefig('pooled_multiclass_mcc.png')


if False:
    df = pd.read_csv('remainder_stratified_crossval_bert.csv', index_col=0)
    bins = df.reset_index()['index'].str.split('_', expand=True)[1]
    df['bin'] = list(bins)
    bert = df.loc[df.index.str.contains('mcc2')].groupby('bin').mean().mean(axis=1)

    df = pd.read_csv('remainder_stratified_crossval_sp5.csv', index_col=0)
    bins = df.reset_index()['index'].str.split('_', expand=True)[1]
    df['bin'] = list(bins)
    sp5 = df.loc[df.index.str.contains('mcc2')].groupby('bin').mean().mean(axis=1)


    df = pd.DataFrame([sp5, bert],index=['SignalP 5.0', 'SignalP 6.0']).T
    df.index = [f'{x} %' for x in range(0,100,10)]
    df.plot(kind='bar', figsize=(6,4))
    plt.legend(loc='lower right')
    plt.ylabel('MCC2')
    plt.xlabel('Maximum identity to training set')
    plt.tight_layout()
    plt.savefig('mcc2.png')