'''
We could reuse the benchmark set of SignalP5.0, but there are three problems
a) partition memberships were changed
b) some sequences might have been dropped
c) reclassification of gram or tat

--> extract all entries that are in the SP5 benchmark set from the finished
SP6 training set. 
'''
import pandas as pd


ORIGINAL_DATA = '../signalp_original_data/benchmark_set.fasta'
NEW_DATA = 'signalp_6_train_set.fasta'
OUT_FILE = 'signalp_5_benchmark_set_updated.fasta'


with open(ORIGINAL_DATA, 'r') as f:
    lines = f.read().splitlines()
    identifiers = lines[::3]
    sequences = lines[1::3]
    labels = lines[2::3]

entry_ids = [x.strip('>').split('|')[0] for x in identifiers]


with open(NEW_DATA, 'r') as f:
    lines = f.read().splitlines()
    identifiers = lines[::3]
    sequences = lines[1::3]
    labels = lines[2::3]

accids = [x.strip('>').split('|')[0] for x in identifiers]

df_trainset =  pd.DataFrame.from_dict({'acc': accids, 'label':labels, 'sequence':sequences,'header':identifiers})
df_trainset = df_trainset.set_index('acc')


failed_entries = 0
with open(OUT_FILE,'w') as f:
    for entry_id in entry_ids:

        try:
            header = df_trainset['header'].loc[entry_id] 
            sequence = df_trainset['sequence'].loc[entry_id]
            label = df_trainset['label'].loc[entry_id]

            f.write(header+'\n')
            f.write(sequence + '\n')
            f.write(label + '\n')

        except KeyError:
            failed_entries += 1
            print(f'Could not find {entry_id}')

print(f'Complete. Original benchmark set: {len(entry_ids)} samples. {failed_entries} samples removed, are not in the train set anymore.')

