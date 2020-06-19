# Data downloading and preprocessing

# 04/06/2020


### Pipeline to process UniRef50
#TODO UniRef50 doesn't work, because you get the reference seq which is not necessarily from the organism
for the sake of sanity ignore fragmentation status, also don't see how it really matters for next token prediction
why not also ignore predicted/observed, apart from heuristic reasons this should not affect clusters, only cluster sizes

- download_data.py:  
  queries uniprot for .fasta files
- mmseqs_clustering_pipeline.sh:  
 clusters a fasta file
- homology_partition_data_split.py:  
uses clustering result to create train, val, test splits and converts to .csv and .lmdb

#### Open
Baseline datasets, both query and how to combine with normal dataset
Preferably the leave plasmodium out approach, and then recombine after splits

Shell script to run it all at once