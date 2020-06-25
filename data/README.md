# Data downloading and preprocessing

## Uniprot Query

 - very big file, use wget to download. python requests cannot keep connection open.

## Clustering

    awk -v 'FS=\t' 'NR>1 {print ">sp|"$1"|"$2"|"$4" "$5" "$6 "\n"$7}' uniprot_dump_16062020.tsv > /work3/felteu/uniprot_eukarya.fasta

- turn uniprot .tsv file into a .fasta file with above command.
- run mmseqs easy-cluster on the fasta file.

## Homology partitioning and saving

- run homology_partition.py on cluster output and uniprot table.
- returns splits, for each {split}-{species} saves .tsv of original table, .txt of sequences only, and .mdf5 of tokenized concatenated sequences.


#### Reason for hdf5: Cannot load full data into memory, too big. Solution: Virtual indexing, implemented in hd5 Dataloader
- data is tokenized and concatenated, saved as hdf5 (takes 100gb ram, but only needs to be done once). 1D array of length total_seq_len.
- calculate tokens_per_batch from total_seq_len/batch_size.
- specify offsets for the batch_size batches into the 1D array. batch i starts at tokens_per_batch*i (batch enumeration starting at 0).
- compute bptt lenghts as before, starting at 0
- when accessing an element, add offsets to start:end, and build indexing matrix. This yields a (bptt,batch_size) array, each batch starting at the correct position.