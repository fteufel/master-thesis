# Identify more organisms with strange SPs

## Data

all_manual_sps.tsv:  
`fragment:no annotation:(type:signal) AND reviewed:yes`

all_manual_sps_class_familiy.tsv:  
Same query, but more taxonomy columns. Queried 18/10/2020

## Strategy

- Extract the SP sequence from the downloaded data
- Use FAMSA to get a distance matrix