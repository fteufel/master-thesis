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



## Putting everything together, 16/11/2020

```
awk "NR%3>0 {print $1}" all_extensions.fasta >all_extensions_seqonly.fasta

cat full_updated_data_seqs_only.fasta update_tat_sp_lipo/all_extensions_seqonly.fasta >/work3/felteu/signalp6_seqonly_for_graphpart.fasta
```
Remaining duplicates for unclear reasons:
`grep ">" signalp6_for_graphpart.fasta | cut -f1 -d"|" | sort | uniq -c | sort -r | head -100`
      2 >Q9VNB5
      2 >Q8TMG0
      2 >Q8R6L9
      2 >Q8DIQ1
      2 >Q7K4Y6
      2 >Q72CX3
      2 >Q6LZY8
      2 >Q186B7
      2 >P76446
      2 >P76445
      2 >P69937
      2 >P39396
      2 >P38878
      2 >P14672
      2 >P14142
      2 >P0AD14
      2 >P0AB58
      2 >P03960
      2 >O59010
      2 >O29867
      2 >F5L478
      2 >D6R8X8
      2 >A0A0H2VG78

Removed manually from all_extensions.fasta. Don't want to figure out how it happened. Looks like a problem with outdated name in the TOPDB dump.

run get_edgelist.py on this set. Can still remove short sequences after this if I want to.
Then, rework make_signalp6_data.py to integrate additional .fasta file.


sequences prepared with `cat full_updated_data_seqs_only.fasta update_tat_sp_lipo/all_extensions_seqonly.fasta >signalp_6_seqs_only.fasta`.  
Relabeled according to  graph-part input requirements with `label=` and no `|` between kingdom and type.

used vs code regex ctrl+h  
```
\|([A-Z_]*)$
--> $1

\|
--> |label=
```

manually removed sequences containing X,B,U,Z from seqs_for_graph_part.fasta

manually removed SPs that don't start with M - those are truncated at n-terminal `^[^M>][A-Z]+`