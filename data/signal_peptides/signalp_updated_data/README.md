# Update the SignalP 5.0 dataset

### revisit definition of gram positive: all monoderm bacteria

Phylum           | ID    |
-----------------|-------|
Firmicutes       | 1239
Actinobacteria   | 201174
Thermotogae      | 200918
Chloroflexi      | 200795
Saccharibacteria |  95818
Tenericutes      | 544448
  
Need to download phylum ID for each record in the SignalP dataset and update those that are not gram-positive yet.

`phylum_ids.tsv` has the phylum ids, apart from those acc ids that are not matchable to uniprot.

```
df = pd.read_csv('signalp_original_data/accids.txt', names = ['Entry']) 
df_phylum = pd.read_csv('signalp_updated_data/phylum_ids.tsv', sep='\t')
df = df.merge(df_phylum, on='Entry', how='left') 

df.loc[df['Taxonomic lineage (PHYLUM)'].isna()]

P92192
P03989
P01892
P01891
Q04826
P04229
P15570
Q44975
G2JM62
```
None of those are gram-positive in Uniparc. Can ignore.

```
gram_pos = df.loc[df['Taxonomic lineage (PHYLUM)'].isin(['Firmicutes', 'Actinobacteria','Thermotogae','Chloroflexi','Saccharibacteria','Tenericutes'])]

ds = LargeCRFPartitionDataset('signalp_original_data/train_set.fasta', return_kingdom_ids=True)
df_ds =  pd.DataFrame.from_dict({'ID':ds.identifiers, 'kingdom':ds.kingdom_ids})
df_ds[['Entry','kingdom','type','partition']] = df_ds['ID'].str.lstrip('>').str.split('|', expand=True)

gram_pos = gram_pos.merge(df_ds, on='Entry')
gram_pos.loc[gram_pos['kingdom']!='POSITIVE']
miscategorized = gram_pos.loc[gram_pos['kingdom']!='POSITIVE']['Entry'].values
```

94 additional sequences are gram positive according to the new rules.

```
with open('train_data_updated.fasta', 'w') as f:
     for idx, ident in enumerate(ds.identifiers):
        acc, kingdom, typ, part = ident.lstrip('>').split('|')
        if acc in miscategorized:
            print(f'updating {acc}')
            kingdom = 'POSITIVE'
        f.write('>'+acc+'|'+kingdom+'|'+typ+'|'+part+'\n')
        f.write(ds.sequences[idx])
        f.write('\n')
        f.write(ds.labels[idx])
        f.write('\n')

```
  
### SPIII data
UniProt release 2020_05

How to tag rest of sequence: O,M,I tags: After SP end: O, then M if annotation in Uniprot. After M, I

#### Bacteria
 https://prosite.expasy.org/PDOC00342
 https://prosite.expasy.org/PS00409
database:(type:prosite ps00409) AND reviewed:yes

Take true positive and false negative.

#### Archaea



### TAT-Lipoproteins
https://onlinelibrary.wiley.com/doi/full/10.1111/j.1365-2958.2007.06034.x