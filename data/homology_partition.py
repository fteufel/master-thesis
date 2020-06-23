import numpy as np
import pandas  as pd
import  argparse
import pandas as pd
import os
import logging
import pickle

def partition_assignment(cluster_vector : np.array, kingdom_vector: np.array, label_vector: np.array, n_partitions: int, n_class: int, n_kingdoms: int) -> np.array:
    ''' Function to separate proteins into N partitions with balanced classes
        Inputs:
            cluster_vector: (n_sequences) integer vector of the cluster id of each sequence
            kingdom_vector: (n_sequences) integer vector of the taxonomy group of each sequence
            label_vector  : (n_sequences) integer vector of another balancing criterion
        Returns:
            (n_sequences) split indicator vector
    '''
    
    # Unique cluster number
    u_cluster = np.unique(cluster_vector)
    
    # Initialize matrices
    loc_number = np.ones((n_partitions,n_kingdoms,n_class))
    cl_number = np.zeros(cluster_vector.shape[0])
    
    processed_clusters = 1
    n_clusters = u_cluster.shape[0]
    for i in u_cluster:
        logger.info(f'Processing cluster {processed_clusters}/{n_clusters}')
        processed_clusters +=1
        # Extract the labels for the proteins in that cluster
        positions = np.where(cluster_vector == i)
        cl_labels = label_vector[positions]
        cl_kingdom = kingdom_vector[positions]
        cl_all = np.column_stack((cl_kingdom,cl_labels))
 
        # Count number of each class
        u, count = np.unique(cl_all, axis=0, return_counts=True)
        
        temp_loc_number = np.copy(loc_number)
        temp_loc_number[:,u[:,0],u[:,1]] += count
        loc_per = loc_number/temp_loc_number
        best_group = np.argmin(np.sum(np.sum(loc_per, axis=2),axis=1))
        loc_number[best_group,u[:,0],u[:,1]] += count
        
        # Store the selected partition
        cl_number[positions] = best_group
    
    print(loc_number.astype(np.int32)-np.ones(loc_number.shape))
    return cl_number



parser = argparse.ArgumentParser(description='Balanced train-test-val split')
parser.add_argument('--cluster-data', type=str, 
                    help='mmseqs2 clustering output .tsv')
parser.add_argument('--uniprot-data', type=str,
                    help = 'the uniprot table')
parser.add_argument('--output-dir', type=str,
                    help = 'dir to save outputs')
args = parser.parse_args()

#to lower the memory burden (and avoid errors when loading from .txt, that are impossible to trace back in this file)
#I applied cut-f1,5 cut -f1,6 to get taxon and length data.
'''
#load - all identifiers are still strings, not integers!
cluster_vector  =  np.genfromtxt(args.cluster_data, dtype = str, delimiter= '\t')
print('clustering data loaded')
#data = np.loadtxt(args.uniprot_table, dtype = str, delimiter= '\t', skiprows= 1)
taxonomy_array =  np.genfromtxt(args.taxon_data, dtype = str, delimiter= '\t', skip_header= 1)
taxonomy_vector = taxonomy_array[:,1]
sequence_ids = taxonomy_array[:,0]
print('taxonomy data loaded')
length_vector =  np.genfromtxt(args.length_data, dtype = str, delimiter= '\t', skip_header= 1)[:,1]
print('sequence length data loaded')
length_vector =  length_vector.astype(int)

print('shapes before further preprocessing')
print(taxonomy_vector.shape)
print(length_vector.shape)
print(cluster_vector.shape)

print('rearranging cluster vector...')
#cluster data is NOT in the original order. fix that:
ids_sorted = np.argsort(sequence_ids)
cluster_pos = np.searchsorted(sequence_ids[ids_sorted], cluster_vector[:,1])
indices = ids_sorted[cluster_pos]


xsorted = np.argsort(cluster_vector[:,1]) #seq IDs are in column 2
ypos = np.searchsorted(cluster_vector[:,1][xsorted], taxonomy_vector) 
indices = xsorted[ypos] # indices of the values of xsorted in taxonomy_vector
cluster_vector = cluster_vector[indices,0] #now seq IDs are in same order as in taxonomy_vector


print('binning lengths...')
hist, bin_edges = np.histogram(length_vector, bins = 50) #might be too much
length_vector_binned = np.digitize(length_vector, bin_edges)

n_classes = np.unique(length_vector_binned).shape[0]
n_taxons = np.unique(taxonomy_vector.shape)[0] #np.unique(taxonomy_vector.shape[0])

print('converting categories to integers...')
#convert  string identifiers to integer categorical identifiers
u, cluster_vector = np.unique(cluster_vector, return_inverse = True)
#u, taxonomy_vector = np.unique(taxonomy_vector, return_inverse = True)
taxonomy_vector = np.where(taxonomy_vector=='Plasmodium',1,0)

print(taxonomy_vector.dtype)
print(length_vector_binned.dtype)
print(cluster_vector.dtype)
print(taxonomy_vector.shape)
print(length_vector_binned.shape)
print(cluster_vector.shape)
print('Partitioning...')
'''

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



####the clean way - let's hope memory can handle it
logger.info('loading UniProt table')
df_seqs = pd.read_csv(args.uniprot_data, sep = '\t')
df_seqs = df_seqs.sort_values('Entry')

logger.info('loading clustering data')
df_cl = pd.read_csv(args.cluster_data, sep = '\t', header = None)
df_cl = df_cl.sort_values(1)

logger.info('creating vectors')
cluster_vector = df_cl[0].astype('category').cat.codes.to_numpy()
taxonomy_vector= df_seqs['Taxonomic lineage (GENUS)']
taxonomy_vector.loc[~(taxonomy_vector=='Plasmodium')] = 'Not Plasmodium'
#taxonomy_vector.loc[ taxonomy_vector=='Plasmodium'] = 1
taxonomy_vector = taxonomy_vector.astype('category').cat.codes
taxonomy_vector = taxonomy_vector.to_numpy()
length_vector = pd.cut(df_seqs['Length'], 50).cat.codes.to_numpy()

n_classes = df_cl[0].astype('category').cat.categories.shape[0]
n_taxons = 2

#I hope this will help me eventually
logger.info('Pickling vectors...')
pickle.dump( {'cluster' : cluster_vector, 'taxon': taxonomy_vector, 'length': length_vector}, open( os.path.join(args.output_dir, 'vectors.pkl'), "wb"  ))


logger.info('Partitioning...')
partition_assignments =  partition_assignment(cluster_vector, taxonomy_vector, length_vector, 10, n_classes, n_taxons)

logger.info(partition_assignments.shape)

pickle.dump( partition_assignments, open( os.path.join(args.output_dir, 'assignments.pkl'), "wb" ) )
#embed()
#np.savetxt('partitions.txt', (cluster_vector, partition_assignments))

#made 10 splits - recombine to get the 3 splits i need
partitions = np.unique(partition_assignments)
train = partitions[:6]
test  = partitions[6:9]
val   = partitions[9:]
train_idx =  np.isin(partition_assignments, train)
test_idx =  np.isin(partition_assignments, test)
val_idx =  np.isin(partition_assignments, val)

train_df = df_seqs.loc[train_idx]
test_df = df_seqs.loc[test_idx]
val_df = df_seqs.loc[val_idx]


train_df.to_csv(os.path.join(args.output_dir, 'train.csv'))
test_df.to_csv(os.path.join(args.output_dir, 'test.csv'))
val_df.to_csv(os.path.join(args.output_dir, 'val.csv'))

len_stats = pd.DataFrame([train_df.describe()['Length'], val_df.describe()['Length'], test_df.describe()['Length']], index = ['Train','Validation', 'Test'])

train_df['Taxonomic lineage (GENUS)'].value_counts()[0]

percent_plasmodium = [df['Taxonomic lineage (GENUS)'].value_counts()[1]/df['Taxonomic lineage (GENUS)'].value_counts()[0] *100 for df in [train_df, test_df, val_df]]
logger.info('train test val Plasmodium %')
logger.info(percent_plasmodium)

