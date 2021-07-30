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

fasta_file = 'signalp_6_train_set.fasta'#'train_set_sequences_only.fasta' #grep -A 1 '>' train_set.fasta | grep -v '\-\-' >train_set_sequences_only.fasta
homology_threshold = 0.2
ggs = path.expanduser('/work3/felteu/fasta36/bin/ggsearch36')

print_hits = True #print each sequence that hits above threshold. Enable if you can't believe the final results.


## make temporary fasta files for each partition and count their sizes
with open(fasta_file, 'r') as f:
        lines = f.read().splitlines()
        identifiers = lines[::3]
        sequences = lines[1::3]
        labels = lines[2::3]

for partition in [0,1,2]:
    for spclass in ['NO_SP','SP','LIPO','TAT','TATLIPO','PILIN']:

        with open(f'partition_{partition}_{spclass}.tmp', 'w') as f:

            for header,sequence in zip(identifiers, sequences):
                if header.split('|')[-1] == str(partition) and header.split('|')[-2] == spclass:
                    f.write(header + '\n')
                    f.write(sequence + '\n')



def count_number_of_violating_seqs():

    ## For each partition, align sequences to all other partitions and count hits that exceed threshold
    ## ensure that if a sequence has multiple hits above threshold in another partition, we still only count 1
    ## (we want to count sequences per partition that have overlap, not the intensity of the overlap per sequence)

    for spclass in ['SP','LIPO','TAT','TATLIPO','PILIN', 'NO_SP']:
        for partition_1 in range(0,3):
            for partition_2 in range(0,3):
                critical_identities = []

                if True: #sanity check#partition_1  != partition_2:

                    with subprocess.Popen(
                        [ggs, '-E', '20758', f'partition_{partition_1}_{spclass}.tmp', f'partition_{partition_2}_{spclass}.tmp'],
                        stdout = subprocess.PIPE,
                        bufsize=1,
                        universal_newlines=True) as proc:


                        last_processed_query = None #this is here so that each query gets only counted once if it has a match
                        query_identities = []

                        # parse the ggsearch output - taken from Magnus
                        for line_nr, line in enumerate(proc.stdout):
                            if '>>>' in line:
                                qry_nr = int(line[2])
                                this_qry = line[6:70].split()[0]#.split('|')[0:4]

                            elif line[0:2] == '>>':
                                this_lib = line[2:66].split()[0]#.split('|')[0]

                            elif line[:13] == 'global/global':
                                identity = float(line.split()[4][:-1])/100

                            
                                ## Gather the parsed results for each query and find maximum threshold violation
                                #if last_processed_query is None: #this is to make it work at the first query
                                #    last_processed_query = this_qry

                                # if query the same as last one, add new identity to list
                                #if this_qry == last_processed_query:
                                #    query_identities.append(identity)

                                # if we proceed to a new query, find max identity of last query and append to critical_identies
                                # update last_processed_query to new query
                                #else:
                                    #if max(query_identities) > homology_threshold:
                                #    critical_identities.append(max(query_identities))
                                #    last_processed_query = this_qry
                                #    query_identities = []
                                if identity > 0.3:
                                    print(f'{spclass}_{partition_1}_{partition_2}: {this_qry}-{this_lib}:{identity}')


                    #critical_identities = np.array(critical_identities)
                    #print(f'{spclass} Partitions {partition_1}-{partition_2}'
                    #    + f' average exceeding identity {critical_identities.mean()}, median {np.median(critical_identities)}, max {critical_identities.max()}, min {critical_identities.min()}'
                    #    )


    df = pd.DataFrame.from_dict(violating_sequence_counts)
    print('Alignments complete.')
    print(df)
    #df.to_csv('partition_overlap_sequence_counts.csv')

    return df


count_number_of_violating_seqs()

similarity_matrices = get_all_similarities()
#save as pickle for convenience. Do nice plots in notebook, not on hpc.
with open('similarity_matrices_partitions.pkl', 'wb') as f:
    pickle.dump(similarity_matrices,f)



#clean up alignment temp files
for partition in [0,1,2,3,4]:
    for spclass in ['NO_SP','SP','LIPO','TAT','TATLIPO','PILIN']:
        remove(f'partition_{partition}_{spclass}.tmp')



