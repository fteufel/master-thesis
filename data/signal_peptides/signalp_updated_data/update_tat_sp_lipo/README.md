# Update sequences new to Uniprot since SignalP5 training set creation

## Queries

submit all IDs in the dataset extended by LIPOTAT and PILIN to Uniprot to get a list that can be used for querying.  
IDs are in `all_old_acc_ids.txt`.


### TAT
`NOT yourlist:M20201116216DA2B77BFBD2E6699CA9B6D1C41EB2046C31H database:(type:prosite ps51318) AND reviewed:yes` > tat_uniprot_prosite_matches.tsv  
201 sequences

`NOT yourlist:M20201116216DA2B77BFBD2E6699CA9B6D1C41EB2046C31H annotation:(type:signal tat-type) NOT database:(type:prosite ps51318)` > tat_uniprot_non_prosite_matches.tsv  
8 sequences

**Directly from Prosite:**  

Manually submit false negatives IDs to Uniprot to get list
`NOT yourlist:M20201116216DA2B77BFBD2E6699CA9B6D1C41EB2046C31H yourlist:M20201116A94466D2655679D1FD8953E075198DA8128A60K` > tat_false_negatives.tsv   

Manually submit unknown IDs to Uniprot to get list
`NOT yourlist:M20201116216DA2B77BFBD2E6699CA9B6D1C41EB2046C31H yourlist:M20201116A94466D2655679D1FD8953E075198DA8128AA7K` > tat_unknown.tsv  

All false negatives are covered by the non-prosite query. file can be ignored.

### LIPO

`NOT yourlist:M20201116216DA2B77BFBD2E6699CA9B6D1C41EB2046C31H database:(type:prosite ps51257) AND reviewed:yes` > lipo_uniprot_prosite_matches.tsv  
812 sequences  


`NOT yourlist:M20201116216DA2B77BFBD2E6699CA9B6D1C41EB2046C31H yourlist:M20201116A94466D2655679D1FD8953E075198DA8128B25R` > lipo_false_negatives.tsv

`NOT yourlist:M20201116216DA2B77BFBD2E6699CA9B6D1C41EB2046C31H yourlist:M20201116A94466D2655679D1FD8953E075198DA8128B5E6` > lipo_unknown.tsv  


### SP

`NOT yourlist:M20201116216DA2B77BFBD2E6699CA9B6D1C41EB2046C31H annotation:(type:signal evidence:experimental) AND reviewed:yes` > sp_uniprot_hits.tsv  
387 sequences 


### NO_SP

#### Soluble
NOT yourlist:M20201116216DA2B77BFBD2E6699CA9B6D1C41EB2046C31H OR locations:(location:"Cytosol [SL-0091]" evidence:experimental) OR locations:(location:"Nucleus [SL-0191]" evidence:experimental) OR locations:(location:"Mitochondrion [SL-0173]" evidence:experimental) OR locations:(location:"Peroxisome [SL-0204]" evidence:experimental) OR locations:(location:"Plastid [SL-0209]" evidence:experimental) NOT annotation:(type:signal)

#### TM
TOPDB pipeline. Resulting Entry names were queried in uniprot, yielding tm_not_in_signalp_uniprot.tsv .   
Get kingdom and sequence from there, make label from topdb data.

## Processing

### Selection
from each match file, the ones that are in unknown are removed.  

from SP query, hits that are also in TAT and LIPO need to be removed (can't differentiate types in uniprot query itself)


### Cleavage sites

SP is not always annotated in Uniprot when a prosite profile matches.

- When annotated, use Uniprot annotation (in the uniprot dump file)
- When not annotated, use Scanprosite output (tat_scanprosite_prosite_matches.tsv)



### Check for TATLIPO candidates

Predict TAT set with lipo only model
C8WLM1 | Subcellular location cell membrane, peripheral membrane protein, literature reference
P18190 | Subcellular location cell membrane, peripheral membrane protein, no evidence
P40120 | periplasmic
Q9AD93 | peptidoglycan anchored
P39660 | substrate binding part of ABC transporter, can't find lipobox manually
P0AAK7 | no evidence
Q8Z765 | periplasmic
P0A1C6 | periplasmic
Q8CW34 | periplasmic
Q8ZPB3 | periplasmic
Q8X9V1 | periplasmic
B5R535 | periplasmic
B5BJ86 | periplasmic
A1AB32 | periplasmic
Q1RBZ1 | periplasmic
Q5PHT4 | periplasmic
Q3Z1F4 | periplasmic
B6IAG9 | periplasmic
B1XDD6 | periplasmic
Q320I8 | periplasmic
A7ZLL8 | periplasmic
B2U143 | periplasmic
C4ZVG4 | periplasmic
A0KYQ9 | no localization info
A3D6B7 | no localization info
P0AAK8 | no localization info
B0R5Y3 | no localization info