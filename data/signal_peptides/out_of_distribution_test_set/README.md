# Build OOD test sets from Uniprot

The process here is based on the one test in ../identify_more_species. 
We need to repeat it, because we have a new training set and new gram-negative/positive classifications.  

Additionally, a negative set would be nice to have.


**Queries**
```
NOT yourlist:M20201203A94466D2655679D1FD8953E075198DA81BC088Q fragment:no annotation:(type:signal evidence:manual)
```
Split this query for kingdoms:  


```
NOT yourlist:M20201203A94466D2655679D1FD8953E075198DA81BC088Q fragment:no annotation:(type:signal evidence:manual) taxonomy:"Eukaryota [2759]"

NOT yourlist:M20201203A94466D2655679D1FD8953E075198DA81BC088Q fragment:no annotation:(type:signal evidence:manual) taxonomy:"Archaea [2157]"

NOT yourlist:M20201203A94466D2655679D1FD8953E075198DA81BC088Q fragment:no annotation:(type:signal evidence:manual) taxonomy:"Bacteria [2]" NOT taxonomy:"Firmicutes [1239]" NOT taxonomy:"Tenericutes [544448]" NOT taxonomy:"Actinobacteria [201174]" NOT taxonomy:"Thermotogae [200918]" NOT taxonomy:"Chloroflexi [200795]" NOT taxonomy:"Candidatus Saccharibacteria [95818]"

(taxonomy:"Firmicutes [1239]" OR taxonomy:"Tenericutes [544448]" OR taxonomy:"Actinobacteria [201174]" OR taxonomy:"Thermotogae [200918]" OR taxonomy:"Chloroflexi [200795]" OR taxonomy:"Candidatus Saccharibacteria [95818]") NOT yourlist:M20201203A94466D2655679D1FD8953E075198DA81BC088Q fragment:no annotation:(type:signal evidence:manual)

```

**Prepare files for alignment**
```
python3 ../identify_more_species/preprocess_for_alignment.py --input_file ../signalp_updated_data/signalp_6_train_set.fasta --output_file 'sp6_train_set_for_alignment.fasta'

python3 ../identify_more_species/sp_fasta_from_tsv.py archaea_manual_sps.tsv eukarya_manual_sps.tsv negative_manual_sps.tsv positive_manual_sps.tsv
```

**Align**


```
`/work3/felteu/fasta36/bin/ggsearch36 archaea_manual_sps_only.fasta sp6_train_set_for_alignment.fasta -m 8 -b 1 >archaea_aligmment.tsv`
/work3/felteu/fasta36/bin/ggsearch36 eukarya_manual_sp_sps_only.fasta sp6_train_set_for_alignment.fasta -m 8 -b 1 >eukarya_alignment.tsv
```