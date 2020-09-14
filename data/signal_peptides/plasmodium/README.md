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
3. map reference sequences back to UniProt and filter for SPs and Plasmodium (RefSeq not necessarily plasmodium after these steps)
4. choose randomly.  
shuf -n 100 >plasmodium_tm_test_set.tsv`

### Sequences without transmembrane helix

taxonomy:"Plasmodium [5820]" fragment:no NOT annotation:(type:signal) NOT annotation:(type:transmem)

yourlist:M20200913E5A08BB0B2D1C45B0C7BC3B55FD26556B4D00DE taxonomy:"Plasmodium [5820]" NOT annotation:(type:signal) NOT annotation:(type:transmem)

shuf -n 100 plasmodium_no_tm_no_sps.tsv > plasmodium_no_tm_test_set.tsv