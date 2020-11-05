# Identifying TAT-LIPO Proteins


## Approaches
- Match TATLIPO training data to Swissprot
- Run Prosite profiles without postprocessing
- Use SPII BERT-CRF to detect Lipobox in TAT sequences in SignalP dataset



## Manual selection of sequences


#### Prosite hits
Acc    |  Name        | Note                 | Accept? | Found evidence
-------|--------------|----------------------|---------|------
A0KYQ9 |  G1092_SHESA |                      | Yes     | No evidence
A1S4U5 |  GH109_SHEAM | SignalP TAT          | Yes     | No evidence
B0R6I5 |  BASB_HALS3  | SignalP TAT          | Yes     | Note: Probably anchored to the membrane by lipids.
C8WLE3 |  URDA_EGGLE  |                      | Yes     | Other Urocanate reductases are membrane-bound https://onlinelibrary.wiley.com/doi/full/10.1111/mmi.12067
D4GSY9 |  ANTRB_HALVD | SignalP TAT          | Yes     | No literature
O50499 |  NGCE_STRCO  |                      | Yes     | putative lipoprotein http://strepdb.streptomyces.org.uk/cgi-bin/annotation.pl?serial=5989&accession=AL645882&width=900
P0A433 |  OPD_SPHSA   | SignalP TAT          | Yes     | 100% identity Lipoprotein https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4817201/
P0A434 |  OPD_BREDI   |                      | Yes     | Lipoprotein https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4817201/
P31545 |  EFEB_ECOLI  | SignalP TAT          | Yes     | Periplasmic protein, no evidence
P39442 |  HCY_NATPH   | FP TAT/SignalP LIPO  | Yes     | Lipidation annotated. Binds a Cu ion, might be reason for TAT https://www.nature.com/articles/nrmicro2814
P94170 |  CAH_NOSS1   | SignalP TAT          | Yes     | No literature, some CAs are membrane-bound (Interpro IPR018338)
Q0HH61 |  G1092_SHESM |                      | Yes     | No literature
Q0HTG8 |  G1092_SHESR | SignalP TAT          | Yes     | No literature
Q1LBC2 |  Y5695_CUPMC | FP TAT               | No      |
Q31Z86 |  EFEB_SHIBS  |                      | Yes     | Periplasmic protein, no evidence
Q3Z396 |  EFEB_SHISS  | SignalP TAT          | Yes     | Periplasmic protein, no evidence
Q59105 |  NOSZ_CUPNH  | SignalP TAT          | Yes     | part of a membrane-attached complex, no evidence for attachment of NosZ
Q67QZ1 |  NAPA_SYMTH  | SignalP TAT          | Yes     | Note: Membrane-associated.
Q8XAS4 |  EFEB_ECO57  | SignalP TAT          |         | Periplasmic protein, no evidence
P0A5I7 |  BLAC_MYCBO  | FN TAT               | Yes     | membrane-bound betalactamases confirmed in other organisms https://doi.org/10.1099/00221287-144-8-2169, none here
P9WKD2 |  BLAC_MYCTO  | FN TAT               | Yes     | membrane-bound betalactamases confirmed in other organisms https://doi.org/10.1099/00221287-144-8-2169, none here
P9WKD3 |  BLAC_MYCTU  | FN TAT               | Yes     | membrane-bound betalactamases confirmed in other organisms https://doi.org/10.1099/00221287-144-8-2169, none here
P0AD44 |  YFHG_ECOLI  | FN TAT               | Yes     | Lipoprotein http://stepdb.eu/strains/k12/secretome_via_tat
A5U493 |  BLAC_MYCTA  | FN TAT/SignalP TAT   | Yes     | membrane-bound betalactamases confirmed in other organisms https://doi.org/10.1099/00221287-144-8-2169, none here
P96809 |  FHMAD_MYCTU | FN TAT/SignalP TAT   | Yes     |Rv0132c is anchored to the cell envelope.


#### LIPO-Bert-CRF hits
Acc    |  Kingdom     | Class                | Accept? | Found evidence
-------|--------------|----------------------|---------|------
O30078 | ARCHAEA      |TAT                   | No      | suspected to be anchored to membrane by other protein
D4GQ16 | ARCHAEA      |TAT                   | Yes     | periplasmic substrate binding part of ABC transporter (predicted)
P07984 | POSITIVE     |TAT                   | No      | Other endoglucanases are lipoproteins
P9WM45 | POSITIVE     |TAT                   | No      | cell wall protein, not annotated as lipoprotein because negative prediction from LipoP. If correct, SP would be 70 AAs long
P18477 | POSITIVE     |TAT                   | No      | Fimbrial protein https://www.ncbi.nlm.nih.gov/pmc/articles/PMC208884/
A4QDY7 | POSITIVE     |TAT                   | No      | uncharacterized, unreviewed
P18278 | NEGATIVE     |TAT                   | Yes     | subcellular location cell membrane, no lipobox motif found manually
P0AB24 | NEGATIVE     |TAT                   | No      | periplasmic 
P13063 | NEGATIVE     |TAT                   | No      | periplasmic
P73452 | NEGATIVE     |TAT                   | Yes     | Lipidation annotated
Q44292 | NEGATIVE     |TAT                   | Yes     | Lipidation annotated
Q55460 | NEGATIVE     |TAT                   | Yes     | subcellular location membrane
Q5MZ56 | NEGATIVE     |TAT                   | Yes     | substrate binding part of ABC transporter
Q9HPK2 | ARCHAEA      |TAT                   | Yes     | substrate binding part of ABC transporter 
A0QXD9 | POSITIVE     |TAT                   | Yes     | substrate binding part of ABC transporter 
Q9ZBW1 | POSITIVE     |TAT                   | No      | no manual annotations 
A4QA36| POSITIVE      |TAT                   | No      | no manual annotations 
A4QI82| POSITIVE      |TAT                   | No      | no manual annotations 
Q9EWT5| POSITIVE      |TAT                   | No      | no manual annotations 
Q9EWQ0| POSITIVE      |TAT                   | Yes     | substrate binding part of ABC transporter 
B4SRN1| NEGATIVE      |TAT                   | No      | periplasm
P36649| NEGATIVE      |TAT                   | No      | periplasm
P37600| NEGATIVE      |TAT                   | No      | periplasm
P38043| NEGATIVE      |TAT                   | Yes     | substrate binding part of ABC transporter, lipidation annotated
B2FU50| NEGATIVE      |TAT                   | No      | periplasm
A0RQ36| NEGATIVE      |TAT                   | No      | periplasm
A7I3Y7| NEGATIVE      |TAT                   | No      | periplasm
A7MLE5| NEGATIVE      |TAT                   | No      | periplasm
B4F2J0| NEGATIVE      |TAT                   | No      | periplasm
B9KCQ2| NEGATIVE      |TAT                   | No      | periplasm
D3VCR0| NEGATIVE      |TAT                   | No      | periplasm
F4HDA7| NEGATIVE      |TAT                   | No      | periplasm
P0A1C5| NEGATIVE      |TAT                   | No      | periplasm
P16028| NEGATIVE      |TAT                   | No      | periplasm
Q8X947| NEGATIVE      |TAT                   | No      | periplasm

http://stdgen.northwestern.edu/stdgen/bacteria/analysis/abc_transporters.html:  

*The transporter shows a common global organization with three types of molecular components. Typically, it consists of two integral membrane proteins (permeases) each having six transmembrane segments, two peripheral membrane proteins that bind and hydrolyze ATP, and a periplasmic (or lipoprotein) substrate-binding protein*

#TODO change label of P9WM45 back to TAT from TATLIPO
#TODO change label of Q9HPK2, A0QXD9, Q9EWQ0, P38043 to TATLIPO

