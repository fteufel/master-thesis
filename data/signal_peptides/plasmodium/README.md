# Making a test set for Plasmodium Signal Peptides

## Positive
Just download annotated sequences, take all.

## Negative

### Sequences with transmembrane helix

1. retrieve IDs of sequences with TM helix in first 70 pos
'''
SELECT ?begin ?name
WHERE 
{
	?protein a up:Protein .
	?protein up:annotation ?annotation .
    ?protein up:mnemonic ?name .
	?annotation a up:Transmembrane_Annotation .
	?annotation up:range ?range .
    ?protein up:organism ?organism .
    ?organism rdfs:subClassOf taxon:5820 .
	?range faldo:begin/faldo:position ?begin .
      FILTER(?begin<70)
}
'''
2. map those to UniRef50
3. filter UniRef50 result for Plasmodium in `common taxon`
4. choose randomly.  
`grep 'Plasmodium' plasmodium_tm_clusters_uniref.tsv | shuf -n 100 >plasmodium_tm_test_set.tsv`