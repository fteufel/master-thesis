# Data downloading and preprocessing

## Uniprot Query
https://www.uniprot.org/uniprot/?query=taxonomy%3A%22Eukaryota+%5B2759%5D%22+fragment%3Ano&columns=id%2Centry%20name%2Cgenes%2Corganism%2Clength%2Clineage(GENUS)%2Csequence
 - very big file, use wget to download. python requests cannot keep connection open.

## Clustering

    awk -v 'FS=\t' 'NR>1 {print ">sp|"$1"|"$2"|"$4" "$5" "$6 "\n"$7}' uniprot_dump_16062020.tsv > /work3/felteu/uniprot_eukarya.fasta

- turn uniprot .tsv file into a .fasta file with above command.
- run mmseqs easy-cluster on the fasta file.

## Homology partitioning and saving

- run homology_partition.py on cluster output and uniprot table.
- returns splits, for each {split}-{species} saves .tsv of original table, .txt of sequences only, and .mdf5 of tokenized concatenated sequences.

- `homology_partition.py` will easily go out of memory at the .hdf5 dump step. Use `make_oneline_hdf5.py` to do just this step.

#### Reason for hdf5: Cannot load full data into memory, too big. Solution: Virtual indexing, implemented in hdf5 Dataloader
- data is tokenized and concatenated, saved as hdf5 (takes 100gb ram, but only needs to be done once). 1D array of length total_seq_len.
- calculate tokens_per_batch from total_seq_len/batch_size.
- specify offsets for the batch_size batches into the 1D array. batch i starts at tokens_per_batch*i (batch enumeration starting at 0).
- compute bptt lenghts as before, starting at 0
- when accessing an element, add offsets to start:end, and build indexing matrix. This yields a (bptt,batch_size) array, each batch starting at the correct position.


- Addition: also creates starting_idx vector, so that all sequences can be accessed directly in the concatenated array by a lookup into this vector.


## Uniref query
https://www.uniprot.org/uniref/?query=taxonomy%3A%22Eukaryota%20(9EUKA)%20%5B2759%5D%22%20identity%3A0.9&columns=id%2Cname%2Ccount%2Clength%2Cidentity%2Csequence%2Ccommon%20taxon%2Creviewed&sort=score&format=tab

Necessary to load genus identifiers manually, because Uniref doesn't have these (are in Uniprot)

__Result__:  
It is hard to query Genus from UniRef, as most identifiers point to UniParc annd not to Uniprot, and there the information is not registered. 
Already querying UniProt for the IDs takes some hours (with multiprocessing), would have to query UniParc to get RefSeq ID for sequences, and then query RefSeq for genus.  

--> Just cluster my original UniProt dump at 90% identity.


1. Cluster UniProt Eukarya dump at 90% identity
2. homology reduce, only take representative sequences
3. Cluster new dataset
4. Homology partition


### 22/07/2020

mmseqs appears to be very resource hungry at 90% identity.
-s parameter controls sensitivity, do not provide it to make it auto infer it from min-seq-id
```
bsub -q hpc -n 12 -W 70:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=40GB]" /zhome/1d/8/153438/eukarya_mmseqs_pipeline/mmseqs/bin/mmseqs createdb uniprot_eukarya.fasta uniprot_eukarya

bsub -q hpc -n 12 -W 70:00 -J ${experiment_name} -o /work3/$USER/logs/${experiment_name}/ -e /work3/$USER/logs/${experiment_name}/ -R "span[hosts=1]" -R "rusage[mem=40GB]" /zhome/1d/8/153438/eukarya_mmseqs_pipeline/mmseqs/bin/mmseqs cluster /work3/felteu/uniprot_eukarya_DB/uniprot_eukarya default_clusterres default_tmp --min-seq-id 0.9 --cov-mode 1 -c 0.8 --cluster-mode 2 --max-seqs 2000

```
#### Full pipeline, this seems to finally work.
1. `distributed_default_cluster.sh`
2. `homology_reduce_no_split.py`
3. .tsv to .fasta conversion
4. `distributed_cluster_for_partition.sh`
5. `homology_partition.py`
