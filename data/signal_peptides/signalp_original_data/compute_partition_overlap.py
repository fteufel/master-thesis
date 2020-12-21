'''
There is homology higher than the threshold between the partitions,
because CD-HIT clustering does not enforce a threshold between the partitions.
Rather, it ensures that the homology within a cluster is high.

We need to measure how high this overlap is:
- Align all sequences in a partition to another partition
- Count the number of sequences that have a match higher than the threshold.

'''
import subprocess
import pandas as pd
import numpy as np
from os import path, remove
import pickle

fasta_file = 'train_set.fasta'#'../signalp_updated_data/signalp_6_train_set.fasta'
classes =  ['NO_SP','SP','LIPO','TAT']#], 'TATLIPO','PILIN']
n_partitions = 5
outname = 'signalp'

ggs = path.expanduser('/work3/felteu/fasta36/bin/ggsearch36')


def get_all_similarities_nonredundant(n_partitions=5, classes = ['NO_SP', 'SP', "LIPO", 'TAT'] ):
    '''Treat all partitions together as one graph.
    Get the edges, ignoring edges between different types.

    '''

    edgedict = {}

    for partition_1 in range(n_partitions):
        for partition_2 in range(n_partitions):
            if partition_1 != partition_2:
                for spclass in classes:#, 'TATLIPO', 'PILIN']:
                    print(f'Processing {partition_1}-{partition_2} {spclass}')

                    with subprocess.Popen(
                            [ggs, '-E', '20758', f'partition_{outname}_{partition_1}_{spclass}.tmp', f'partition_{outname}_{partition_2}_{spclass}.tmp'],
                            stdout = subprocess.PIPE,
                            bufsize=1,
                            universal_newlines=True) as proc:


                            # parse the ggsearch output - taken from Magnus
                            for line_nr, line in enumerate(proc.stdout):
                                if '>>>' in line:
                                    qry_nr = int(line[2])
                                    this_qry = line[6:70].split()[0]#.split('|')[0:4]

                                elif line[0:2] == '>>':
                                    this_lib = line[2:66].split()[0]#.split('|')[0]

                                elif line[:13] == 'global/global':
                                    identity = float(line.split()[4][:-1])/100

                                    if this_lib != this_qry: #no edge to self.
                                        # Sets are not hashable, tuples are. sort and make tuple.
                                        key = tuple(sorted([this_lib, this_qry]))
                                        edgedict[key] = identity

    return edgedict



## make temporary fasta files for each partition and count their sizes
with open(fasta_file, 'r') as f:
        lines = f.read().splitlines()
        identifiers = lines[::3]
        sequences = lines[1::3]
        labels = lines[2::3]


for partition in range(n_partitions):
    for spclass in classes:#,'TATLIPO','PILIN']:

        with open(f'partition_{outname}_{partition}_{spclass}.tmp', 'w') as f:

            for header,sequence in zip(identifiers, sequences):
                if header.split('|')[-1] == str(partition) and header.split('|')[-2] == spclass:
                    f.write(header + '\n')
                    f.write(sequence + '\n')


identitydict =  get_all_similarities_nonredundant(n_partitions)

# bin into 100 bins - one for each percent identity. Should be enough.
identitities = np.array(list(identitydict.values())).astype(float)
results, edges = np.histogram(identitities, density=False, bins=100)

with open(f'/work3/felteu/inter_partition_edges_histogram_nonorm_{outname}.pkl', 'wb') as f:
    pickle.dump({'results': results, 'edges': edges}, f)


#save as pickle for convenience. Make nice plots in notebook, not on hpc.
#with open(f'/work3/felteu/inter_partition_edges_{outname}.pkl', 'wb') as f:
#    pickle.dump(identitydict,f)



#clean up alignment temp files
for partition in n_partitions:
    for spclass in classes:
        remove(f'partition_{outname}_{partition}_{spclass}.tmp')