'''
Homology partitioning script for EC prediction.
Also takes care of preprocessing the prediction labels (level 1 of EC code)
0 = no enzyme
'''
import numpy as np
import pandas as pd
import os
import argparse
import logging
import pickle
from typing import List

def setup_logging(dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(dir, 'log.txt'))
    formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%y/%m/%d %H:%M:%S")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger

def partition_assignment_batched(cluster_vector : np.ndarray, kingdom_vector: np.ndarray, label_vector: np.ndarray, n_partitions: int, n_class: int, n_kingdoms: int, bsize = 1) -> np.array:
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
        # First clusters go to split 0.
        #  
        if i <2:
            best_group = 0
        else:
            best_group = np.argmin(np.sum(np.sum(loc_per, axis=2),axis=1))
        loc_number[best_group,u[:,0],u[:,1]] += count
        
        # Store the selected partition
        cl_number[positions] = best_group

        i += bsize
    
    return cl_number


def class_assignments_counts(class_vector: np.ndarray, assignment_vector: np.ndarray) -> np.ndarray:
    ''' Helper function to get a count table.
        Inputs:
            class_vector: (n_sequences) integer vector of the class of each sequence
            assignment_vector: (n_sequences) split indicator vector
        Returns:
            (n_classes, n_partitions) sequence counts for each partition and class
    '''
    n_classes = np.unique(class_vector).shape[0]
    n_partitions = np.unique(assignment_vector).shape[0]
    partition_class_counts = np.zeros((n_classes, n_partitions), dtype = int)
    # 1. get distributions for each partition
    for assignment in np.unique(assignment_vector):
        counts = np.bincount(class_vector[assignment_vector == assignment], minlength = n_classes)
        partition_class_counts[:, int(assignment)] = counts
    
    return partition_class_counts

def iterative_assign_to_split(partition_class_counts: np.ndarray, split_sizes: List[int]) -> List[List[int]]:
    '''Function to assign partitions to splits iteratively, ensuring that all classes are represented in all splits.
    Inputs:
        partition_class_counts: (n_classes, n_partitions) array of counts of sequences for each class and partition
        split_sizes: (n_splits) list of number of partitions to be assigned to each split
    
    NOTE this function is far from complete. It just works for now, by assuring all splits don't have zeros anymore before filling splits up to desired size.
    '''
    split_sizes = np.array(split_sizes)
    assert split_sizes.sum() == partition_class_counts.shape[1], f"Total of split_sizes ({split_sizes.sum()}) needs to match the number of partitions ({partition_class_counts.shape[1]}) in partition_class_counts."
    #assert that no class has less non-zero partitions then there are splits - this would make the task impossible right away
    assert (partition_class_counts !=0).sum(axis =1).min() >= len(split_sizes), 'Class label underrepresentation to severe. Balanced split not possible.'

    split_assignments = [ [] for _ in range(len(split_sizes)) ]
    class_split_counts = np.zeros((partition_class_counts.shape[0], len(split_sizes)))

    #iterate over partitions, assign to splits that still have 0.
    for i in range(partition_class_counts.shape[1]):
        splits_with_zeros = np.where((class_split_counts == 0).any(axis =0))[0] #indices of splits that still have 0 counts.
        splits_not_full = np.where(split_sizes - np.array([len(split) for split in split_assignments]) > 0)[0]

        #NOTE this assumes that non-zero partitions come first in the iteration. Is the case in the current setup, but checking here would be better.
        #Otherwise might assign partition to split so that split is full, but there are still 0s for some classes left.
        if len(splits_with_zeros) > 0:
            best_split = splits_with_zeros[0] #just take the first one.
        else :
            best_split = splits_not_full[0]

        split_assignments[best_split].append(i)
        class_split_counts[:,best_split] = class_split_counts[:,best_split] + partition_class_counts[:, i]

    return split_assignments
    

parser = argparse.ArgumentParser(description='Balanced train-test-val split')
parser.add_argument('--cluster-data', type=str, 
                    help='mmseqs2 clustering output .tsv')
parser.add_argument('--uniprot-data', type=str,
                    help = 'the uniprot table')
parser.add_argument('--output-dir', type=str,
                    help = 'dir to save outputs')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

logger = setup_logging(args.output_dir)

#
# Preprocess Data
#

logger.info('loading UniProt table')
df_seqs = pd.read_csv(args.uniprot_data, sep = '\t')
df_seqs = df_seqs.sort_values('Entry').reset_index(drop = True)

logger.info('Processing raw annotations to classification labels')
df_seqs['target label'] = df_seqs['EC number'].str.split('.', n=1, expand = True)[0].fillna(0).astype('int64')

logger.info('loading clustering data')
df_cl = pd.read_csv(args.cluster_data, sep = '\t', header = None)
df_cl = df_cl.sort_values(1).reset_index(drop = True)

# subsample the negative sequences
n_negative_samples = (df_seqs['target label'] == 0).sum()
n_positive_samples = (df_seqs['target label'] != 0).sum()

negative_indices = np.where(df_seqs['target label'] == 0)

np.random.seed(42)
drop_idx = np.random.choice(negative_indices[0], size =n_negative_samples - n_positive_samples, replace = False)

df_seqs = df_seqs.drop(drop_idx)
df_cl = df_cl.drop(drop_idx)


#NOTE ensure that indexing starts at 0, so numpy indices (returned by partition_assignment() ) behave as expected.
df_seqs = df_seqs.reset_index(drop = True)
df_cl = df_cl.reset_index(drop = True)


logger.info('creating vectors')
cluster_vector = df_cl[0].astype('category').cat.codes.to_numpy()

#class_vector= df_seqs['EC number']
#class_vector.loc[~class_vector.isna()] = 'Enzyme'
#class_vector = class_vector.astype('category').cat.codes
class_vector = df_seqs['target label'] #try full class balancing, not only enzyme=non-enzyme
class_vector = class_vector.to_numpy()
length_vector = pd.cut(df_seqs['Length'], 50).cat.codes.to_numpy()

n_clusters = df_cl[0].astype('category').cat.categories.shape[0]
n_classes = 8

logger.info('Pickling vectors...')
pickle.dump( {'cluster' : cluster_vector, 'class': class_vector, 'length': length_vector}, open( os.path.join(args.output_dir, 'vectors.pkl'), "wb"  ))

n_partitions =10 

if os.path.exists(os.path.join(args.output_dir, 'assignments.pkl')):
    logger.info('Partition assignments found! loading...')
    with open(os.path.join(args.output_dir, 'assignments.pkl'), "rb") as input_file:
        partition_assignments = pickle.load(input_file)
    logger.info('Partition assignments loaded.')
else:
    logger.info('Partitioning...')
    partition_assignments =  partition_assignment_batched(cluster_vector, class_vector, length_vector, n_partitions, n_clusters, n_classes)
    logger.info(partition_assignments.shape)
    pickle.dump( partition_assignments, open( os.path.join(args.output_dir, 'assignments.pkl'), "wb" ) )



#
# Combine homology partitions to train, test and val splits (ratios hardcoded, distribute 10 partitions 6:3:1)
#
#made 10 partitions - recombine to get 3 splits
partitions = np.unique(partition_assignments)
logging.info(partitions)

train_parts = int(n_partitions * 0.6)
test_parts = int(n_partitions * 0.2)
val_parts = int(n_partitions * 0.2)
#TODO assert that ratios work with partition number

# When we have many classes, there is a risk that homology partitioning causes some small size classes to have a count of 0 in some partitions.
# This is not a problem by itself, we just need to make sure that when combining the partitions to splits, that in each split all classes are represented.
# 
partition_class_counts = class_assignments_counts(class_vector, partition_assignments)
#Now, find a combination of partitions that matches the conditions and ensures that there is no 0 count.
split_assignments = iterative_assign_to_split(partition_class_counts, [train_parts, test_parts, val_parts])


train = split_assignments[0] 
test = split_assignments[1]
val = split_assignments[2]
train_idx =  np.isin(partition_assignments, train)
test_idx =  np.isin(partition_assignments, test)
val_idx =  np.isin(partition_assignments, val)



##
## Split original uniprot table and save
##

train_df = df_seqs.loc[train_idx]
test_df = df_seqs.loc[test_idx]
val_df = df_seqs.loc[val_idx]


logger.info(pd.DataFrame([x['target label'].value_counts() for x in [train_df, test_df, val_df]]))

result_dir = os.path.join(args.output_dir, 'splits')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
pd.DataFrame([x['target label'].value_counts() for x in [train_df, test_df, val_df]], index = ['Train', 'Test', 'Valid']).to_csv(os.path.join(result_dir, 'class_stats.csv'))

train_df.to_csv(os.path.join(result_dir, 'train_full.tsv'), sep = '\t')
test_df.to_csv(os.path.join(result_dir, 'test_full.tsv'), sep = '\t')
val_df.to_csv(os.path.join(result_dir, 'valid_full.tsv'), sep = '\t')


