# Identify more organisms with strange SPs

## Data

all_manual_sps.tsv:  
`fragment:no annotation:(type:signal) AND reviewed:yes`

all_manual_sps_class_familiy.tsv:  
Same query, but more taxonomy columns. Queried 18/10/2020

## Strategy

- Extract the SP sequence from the downloaded data
- Use FAMSA to get a distance matrix


## Strategy 26/10/2020

- Perform global alignment between individual sequences of species of interest and signalp training data
- On SP positions only, preprocess training data accordingly

.tsv files created like this:
taxonomy:"Thermotogae [200918]" fragment:no annotation:(type:signal) reviewed:yes





- Prepare SignalP reference database with `preprocess_for_alignment.py`
- Prepare downloaded uniprot .tsvs with `sp_fasta_from_tsv.py`
- Run fasta ggsearch:

`/work3/felteu/fasta36/bin/ggsearch36 archaeoglobi_sps_only.fasta ../signalp_original_data/train_set_for_alignment.fasta -m 8 -b 1`