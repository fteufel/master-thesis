## Queries for the Uniprot SPARQL endpoint


### 1. Count Gram-positive sequences in UniRef100

From SignalP: *Gram-positive bacteria were defined as Firmicutes plus Actinobacteria. We did not include Tenericutes (Mycoplasma and related genera) since they do not seem to have a type I signal peptidase at all*  

-Firmicutes
-Actinobacteria


__Full PHYLUM counts__
```
SELECT (count(*) as ?c)
WHERE
{
  ?protein a up:Protein .
  ?protein up:representativeFor ?cluster .
  ?protein up:organism ?organism.
  ?organism rdfs:subClassOf ?parent .
  ?parent up:rank ?rank .
  
  ?cluster up:identity ?ident .
  	FILTER(?ident=1.0 && ?rank=up:Phylum)
}
GROUP BY ?parent
```

__Filtered, for relevant orgs__

```
SELECT ?parentname (count(*) as ?count)
WHERE
{
  ?protein a up:Protein .
  ?protein up:representativeFor ?cluster .
  ?protein up:organism ?organism.
  ?organism rdfs:subClassOf ?parent .
  ?parent up:rank ?rank .
  ?parent up:scientificName ?parentname.
  
  ?cluster up:identity ?ident .
  FILTER(?ident=1.0 && ?rank=up:Phylum && ?parentname IN ("Actinobacteria", "Firmicutes", "Tenericutes"))
}
GROUP BY ?parentname
```
"Firmicutes"xsd:string	"16241912"xsd:int
"Actinobacteria"xsd:string	"19850266"xsd:int
"Tenericutes"xsd:string	"250954"xsd:int

FILTER(?ident=1.0 && ?rank=up:Phylum) 
150841135



TOTAL UniRef100 (sparql) "       150,841,135"xsd:int
Total UniRef100 (uniprot direct) 235,561,514

Total UniRef100 (sparql without taxon filter) 156 999 503
Count clusters (sparql): 235 561 514


sequences with rank Phylum : 155 901 986

sequences with organism identifier: 156 553 952

__Archaea__
```
SELECT (count(*) as ?count)
WHERE
{
  ?protein a up:Protein .
  ?protein up:representativeFor ?cluster .
  ?protein up:organism ?organism.
  ?organism rdfs:subClassOf ?parent .
  ?parent up:rank ?rank .
  ?parent up:scientificName ?parentname.
  
  ?cluster up:identity ?ident .
  FILTER(?ident=1.0 && ?rank=up:Superkingdom && ?parentname ='Archaea')
}
```
__Result:__ 3 748 640



__All__  

```
SELECT (count(*) as ?count)
WHERE
{
  ?protein a up:Protein .
  ?protein up:representativeFor ?cluster .
  ?protein up:organism ?organism.
  ?organism rdfs:subClassOf ?parent .
  ?parent up:rank ?rank .
  ?parent up:scientificName ?parentname.
  
  ?cluster up:identity ?ident .
  FILTER(?ident=1.0 && ?rank=up:Superkingdom)
}
```