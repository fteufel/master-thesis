# LM training data for parasitic wasps

## Raw data
Data from _Functional insights from the GC-poor genomes of two aphid parasitoids, Aphidius ervi and Lysiphlebus fabarum_:  
https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-020-6764-0  
Downloads:  
https://bipaa.genouest.org/sp/lysiphlebus_fabarum/download/annotation/v1.0/  
https://bipaa.genouest.org/sp/aphidius_ervi/download/annotation/v3.0/  

## Processing

Different fasta headers:  
Uniprot: `>sp|Q0VCA8|3HAO_BOVIN|Bos taurus (Bovine) 286 Bos (oxen, cattle)`  
BIPAA: `>LF000001-PA gene=LF000001`  
Should just work without any problems: _If none of the header supported could be detected than we extract everything from header start (excluding >) until the first whitespace._  

1. New data is appended to the .fasta UniProt Eukarya dump.
```
cat ~/eukarya_mmseqs_pipeline/parasitic_wasps_12082020/aphidius_ervi_OGS3.0_20161222_proteins_2.fa ~/eukarya_mmseqs_pipeline/parasitic_wasps_12082020/lysiphlebus_fabarum_OGS1.0_20170110_proteins_2.fa /work3/felteu/uniprot_eukarya.fasta >concatenated_data.fasta
```
2. `distributed_default_cluster.sh`
3. `distributed_cluster_for_partition.sh` on  clusterRes_rep_seq.fasta
4. `homology_partition.py`

