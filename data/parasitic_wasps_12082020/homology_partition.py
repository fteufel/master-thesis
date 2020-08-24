'''
Homology partitioning script adapted for the parasitic wasps dataset.
Main difference to plasmodium: no .tsv file with taxon identifiers.
Input is .fasta
'''

import numpy as np
import pandas  as pd
import  argparse
import pandas as pd
import os
import sys
import logging
import pickle

sys.path.append('../..')
sys.path.append('/zhome/1d/8/153438/experiments/master-thesis/')
from train_scripts.training_utils import VirtualBatchTruncatedBPTTHdf5Dataset

# Main preprocessing script:
# Inputs:
#   Uniprot table
#   MMSEQS2 Clustering result (clusters.tsv) from this uniprot table.
#   Use this to run mmseqs2 clustering on uniprot tab separated output, as mmseqs wants a  fasta file. assumes sequence in column 7.
#   awk -v 'FS=\t' 'NR>1 {print ">sp|"$1"|"$2"|"$4" "$5" "$6 "\n"$7}' uniprot_dump_16062020.tsv > /work3/felteu/uniprot_eukarya.fasta
# Outputs (1x for eukarya, 1x for plasmodium):
#   .pkl intermediate files, Ordered vectors and partition assignments (length num_sequences)
#   .tsv split files, same format as uniprot table for val, test, train
#   .txt sequence only line-by-line for val, test, train
#   .mdf5 concatenated tokenized sequence data for val, test, train


def partition_assignment_batched(cluster_vector : np.array, kingdom_vector: np.array, label_vector: np.array, n_partitions: int, n_class: int, n_kingdoms: int, bsize = 500) -> np.array:
    ''' Function to separate proteins into N partitions with balanced classes
        Inputs:
            cluster_vector: (n_sequences) integer vector of the cluster id of each sequence
            kingdom_vector: (n_sequences) integer vector of the taxonomy group of each sequence
            label_vector  : (n_sequences) integer vector of another balancing criterion
        Returns:
            (n_sequences) split indicator vector

        Note: Hardcoded that first batch goes to partition 0.
    '''
    
    # Unique cluster number
    u_cluster, sizes = np.unique(cluster_vector, return_counts= True)
    #biggest clusters first
    u_cluster_ordered = u_cluster[np.argsort(sizes)[::-1]]
    
    # Initialize matrices
    loc_number = np.ones((n_partitions,n_kingdoms,n_class))
    cl_number = np.zeros(cluster_vector.shape[0])
    
    n_clusters = u_cluster.shape[0]
    i = 0
    while i < n_clusters:
        if n_clusters - i < bsize:
            current_clusters = u_cluster_ordered[i:]
        else:
            current_clusters = u_cluster_ordered[i:i+bsize]
        if i % 5000 == 0:
            logger.info(f'Processing cluster {i}/{n_clusters}')
        # Extract the labels for the proteins in that cluster
        positions = np.isin(cluster_vector, current_clusters)
        cl_labels = label_vector[positions]
        cl_kingdom = kingdom_vector[positions]
        cl_all = np.column_stack((cl_kingdom,cl_labels))
 
        # Count number of each class
        u, count = np.unique(cl_all, axis=0, return_counts=True)
        
        temp_loc_number = np.copy(loc_number)
        temp_loc_number[:,u[:,0],u[:,1]] += count
        loc_per = loc_number/temp_loc_number
        # There are 533 clusters larger than 2000 seqs in the dataset. Just throw most of them to partition 0.
        # Easy fix, otherwise cannot assure later on that large clusters end up in training split. 
        if i == 0:
            best_group = 0
        else:
            best_group = np.argmin(np.sum(np.sum(loc_per, axis=2),axis=1))
        loc_number[best_group,u[:,0],u[:,1]] += count
        
        # Store the selected partition
        cl_number[positions] = best_group

        i += bsize
    
    print(loc_number.astype(np.int32)-np.ones(loc_number.shape))
    return cl_number

def read_fasta(filepath: str) -> pd.DataFrame:
    '''Read fasta file with mixed headers.
    Can handle two different fasta headers:
    >sp|A0A4W4EHJ5|A0A4W4EHJ5_ELEEL|Electrophorus electricus (Electric eel) (Gymnotus electricus) 1067 Electrophorus
    >AE3000395-PA gene=AE3000395
    '''
    with open(filepath, 'r') as f:
        lines = f.read().splitlines() #f.readlines()
        identifiers = lines[::2]
        sequences = lines[1::2]

    entries = []
    groups = []
    lengths = []
    for i, identifier in enumerate(identifiers):
        lengths.append(len(sequences[i]))
        if identifier.startswith('>sp') or identifier.startswith('>tr'):
            entry = identifier.split('|')[1]
            entries.append(entry)
            groups.append(0)
        
        elif 'gene=' in identifier:
            entry = identifier.split()[0]
            entries.append(entry)
            groups.append(1)
        else:
            raise NotImplementedError(f'No parsing for this type of header defined: {identifier}')


    return pd.DataFrame.from_dict({'Entry': entries, 'Sequence': sequences, 'Group': groups, 'Length': lengths})

parser = argparse.ArgumentParser(description='Balanced train-test-val split')
parser.add_argument('--cluster-data', type=str, 
                    help='mmseqs2 clustering output .tsv')
parser.add_argument('--sequence-data', type=str,
                    help = 'mmseqs2 clustering input .fasta')
parser.add_argument('--output-dir', type=str,
                    help = 'dir to save outputs')
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S")
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

logger.addHandler(c_handler)
logger.addHandler(f_handler)

##
## Load data, homology partition
##

logger.info('loading UniProt table')
df_seqs = read_fasta(args.sequence_data)
df_seqs = df_seqs.sort_values('Entry').reset_index(drop=True)

logger.info('loading clustering data')
df_cl = pd.read_csv(args.cluster_data, sep = '\t', header = None)
df_cl = df_cl.sort_values(1).reset_index(drop=True)

logger.info('creating vectors')
cluster_vector = df_cl[0].astype('category').cat.codes.to_numpy()
#taxonomy_vector= df_seqs['Taxonomic lineage (GENUS)']
#taxonomy_vector.loc[~(taxonomy_vector=='Plasmodium')] = 'Not Plasmodium'
#taxonomy_vector.loc[ taxonomy_vector=='Plasmodium'] = 1
#taxonomy_vector = taxonomy_vector.astype('category').cat.codes
#taxonomy_vector = taxonomy_vector.to_numpy()
taxonomy_vector = df_seqs['Group'].to_numpy()
length_vector = pd.cut(df_seqs['Length'], 50).cat.codes.to_numpy()

n_classes = df_cl[0].astype('category').cat.categories.shape[0]
n_taxons = 2

logger.info('Pickling vectors...')
pickle.dump( {'cluster' : cluster_vector, 'taxon': taxonomy_vector, 'length': length_vector}, open( os.path.join(args.output_dir, 'vectors.pkl'), "wb"  ))

if os.path.exists(os.path.join(args.output_dir, 'assignments.pkl')):
    logger.info('Partition assignments found! loading...')
    with open(os.path.join(args.output_dir, 'assignments.pkl'), "rb") as input_file:
        partition_assignments = pickle.load(input_file)
    logger.info('Partition assignments loaded.')
else:
    logger.info('Partitioning...')
    partition_assignments =  partition_assignment_batched(cluster_vector, taxonomy_vector, length_vector, 10, n_classes, n_taxons)
    logger.info(partition_assignments.shape)
    pickle.dump( partition_assignments, open( os.path.join(args.output_dir, 'assignments.pkl'), "wb" ) )


#
# Combine homology partitions to train, test and val splits (ratios hardcoded, distribute 10 partitions 6:3:1)
#
#made 10 partitions - recombine to get 3 splits
partitions = np.unique(partition_assignments)

reorder_partitions = False #TODO make a real arg somewhere
#ensure that large clusters end up in training split
#This was an ad-hoc solution when i already had partitions. When I already force the large clusters to partition 0, it becomes unnecessary
if reorder_partitions == True:
    cluster_counts = []
    for partition_id in partitions:
        partition_idx = (partition_assignments == partition_id)
        cluster_count = np.unique(cluster_vector[partition_idx]).shape[0]
        cluster_counts.append(cluster_count)
        logger.info(f'Partition {partition_id} - Clusters per seq: {cluster_count/partition_idx.sum():.3f}')
        #assign the ones with lowest cluster count to train split
        partitions_ordered = partitions[np.argsort(cluster_counts)]
else: 
    partitions_ordered = partitions
logging.info(partitions_ordered)

train = partitions_ordered[:6]
test  = partitions_ordered[6:9]
val   = partitions_ordered[9:]
train_idx =  np.isin(partition_assignments, train)
test_idx =  np.isin(partition_assignments, test)
val_idx =  np.isin(partition_assignments, val)

#just for debug info, print top 6 clusters by size for each split
for split, idx in zip(['train', 'test', 'val'],[train_idx, test_idx, val_idx]):
    clustered = cluster_vector[idx]
    clusters, sizes = np.unique(clustered, return_counts= True)
    for i in np.argsort(sizes)[::-1][:6]:
        logger.info(f'Cluster {split}: {clusters[i]} : {sizes[i]} sequences')


##
## Split original uniprot table and save
##

train_df = df_seqs.loc[train_idx]
test_df = df_seqs.loc[test_idx]
val_df = df_seqs.loc[val_idx]

eukarya_dir = os.path.join(args.output_dir, 'eukarya')
if not os.path.exists(eukarya_dir):
    os.mkdir(eukarya_dir)

train_df.to_csv(os.path.join(eukarya_dir, 'eukarya_train_full.tsv'), sep = '\t')
test_df.to_csv(os.path.join(eukarya_dir, 'eukarya_test_full.tsv'), sep = '\t')
val_df.to_csv(os.path.join(eukarya_dir, 'eukarya_val_full.tsv'), sep = '\t')


pd.DataFrame([train_df.describe()['Length'], val_df.describe()['Length'], test_df.describe()['Length']], index = ['Train','Validation', 'Test']).to_csv(os.path.join(args.output_dir, 'len_stats.csv'))


#Plasmodium only split
logger.info('Saving wasp only splits...')
wasp_dir = os.path.join(args.output_dir, 'wasp')
if not os.path.exists(wasp_dir):
    os.mkdir(wasp_dir)
train_df_wasp = train_df.loc[train_df['Group'] == 1]
train_df_wasp.to_csv(os.path.join(wasp_dir, 'wasp_train_full.tsv'), sep ='\t')
test_df_plasm = test_df.loc[test_df['Group'] == 1]
test_df_plasm.to_csv(os.path.join(wasp_dir, 'wasp_test_full.tsv'), sep ='\t')
val_df_plasm = val_df.loc[val_df['Group'] == 1]
val_df_plasm.to_csv(os.path.join(wasp_dir, 'wasp_val_full.tsv'), sep ='\t')


#For convenience - also save sequence only. Can be used by the dataset classes.
logger.info('Saving sequence .txt files')
train_df_plasm['Sequence'].to_csv(os.path.join(wasp_dir, 'train.txt'), header = None, index = None)
test_df_plasm['Sequence'].to_csv(os.path.join(wasp_dir, 'test.txt'), header = None, index = None)
val_df_plasm['Sequence'].to_csv(os.path.join(wasp_dir, 'valid.txt'), header = None, index = None)

train_df['Sequence'].to_csv(os.path.join(eukarya_dir, 'train.txt'), header = None, index = None)
test_df['Sequence'].to_csv(os.path.join(eukarya_dir, 'test.txt'), header = None, index = None)
val_df['Sequence'].to_csv(os.path.join(eukarya_dir, 'valid.txt'), header = None, index = None)

logger.info(f'Train seqs euk: {len(train_df)}')
logger.info(f'Test  seqs euk: {len(test_df)}')
logger.info(f'Valid seqs euk: {len(val_df)}')
logger.info(f'Train seqs pla: {len(train_df_plasm)}')
logger.info(f'Test  seqs pla: {len(test_df_plasm)}')
logger.info(f'Valid seqs pla: {len(val_df_plasm)}')
#
# Make Mdf5 files 
#

logger.info('Creating one-line .hdf5 files...')
dataset = VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_array(train_df['Sequence'], output_file=os.path.join(eukarya_dir, 'train.hdf5'))
logger.info(f'created {dataset.data_file}')
dataset = VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_array(test_df['Sequence'], output_file=os.path.join(eukarya_dir, 'test.hdf5'))
logger.info(f'created {dataset.data_file}')
dataset = VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_array(val_df['Sequence'], output_file=os.path.join(eukarya_dir, 'valid.hdf5'))
logger.info(f'created {dataset.data_file}')
dataset = VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_array(train_df_plasm['Sequence'], output_file=os.path.join(wasp_dir, 'train.hdf5'))
logger.info(f'created {dataset.data_file}')
dataset = VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_array(test_df_plasm['Sequence'], output_file=os.path.join(wasp_dir, 'test.hdf5'))
logger.info(f'created {dataset.data_file}')
dataset = VirtualBatchTruncatedBPTTHdf5Dataset.make_hdf5_from_array(val_df_plasm['Sequence'], output_file=os.path.join(wasp_dir, 'valid.hdf5'))
logger.info(f'created {dataset.data_file}')