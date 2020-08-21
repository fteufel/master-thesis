# Prediction of enzyme class (EC number) in Plasmodium

## Raw data

There are too few enzyme sequences with evidence in Uniprot to train a model. Therefore, also use auto annotated sequences.
While this is of course not the gold standard for training data, it does not invalidate the results in the setup in which the data will be used:

- Data is homology partitioned
- Identical classification models are trained both on the Eukarya LM and the Plasmodium LM
- Should the Plasmodium LM perform better, the "Dialect" (lower perplexity) is indeed meaningful for other tasks.

Beware that for this task, it will probably still perform worse than a model trained on high-quality enzyme sequences across multiple organisms, just because the training data is more powerful there.

__UniProt Query__  

Enzymes: `taxonomy:"Plasmodium [5820]" ec:*`  
https://www.uniprot.org/uniprot/?query=taxonomy%3A%22Plasmodium%20%5B5820%5D%22%20ec%3A*&columns=id%2Centry%20name%2Creviewed%2Corganism%2Clineage(GENUS)%2Cec%2Csequence&sort=score&format=tab

All: `taxonomy:"Plasmodium [5820]" fragment:no`  
https://www.uniprot.org/uniprot/?query=taxonomy%3A%22Plasmodium%20%5B5820%5D%22%20fragment%3Ano&columns=id%2Centry%20name%2Creviewed%2Corganism%2Clineage(GENUS)%2Cec%2Csequence%2Clength&sort=score&format=tab

Go with not only enzymes dataset for now. If needed, can still subset this later on. Homology clustering valid in both cases.

## Processing

1. convert
    awk -v 'FS=\t' 'NR>1 {print ">sp|"$1"|"$2"|"$4" "$5" "$6 "\n"$7}' uniprot_plasm_w_ec.tsv > uniprot_plasm_w_ec.fasta

2. mmseqs clustering
    RUNNER='mpirun' mmseqs easy-cluster /work3/felteu/data/ninety_percent_identity_reduced/reduced_dataset.fasta /work3/felteu/MPI_part_clusterRes MPI_part_tmp -a --min-seq-id 0.4 --threads 1 --cov-mode 1 -c 0.8 --cluster-mode 2 --split-memory-limit 10G
3. homology partition, balanced for non-enzymes