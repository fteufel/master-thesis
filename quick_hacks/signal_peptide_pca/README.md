Recreate PCA plots from plasmodium sp paper, that showed that plasmodium sps are different in their AA composition.
Instead of AA composition, use averaged amino acid embeddings.

Data needed:
- Proteins with SPs from Eukarya
Split into SPs, mature protein, both for pla and euk.

Inital query:
```
https://www.uniprot.org/uniprot/?query=taxonomy%3A%22Eukaryota%20(9EUKA)%20%5B2759%5D%22%20fragment%3Ano%20annotation%3A(type%3Asignal%20evidence%3Amanual)&columns=id%2Centry%20name%2Creviewed%2Corganism%2Clength%2Cprotein%20names%2Csequence%2Clineage(GENUS)%2Cfeature(SIGNAL)&sort=score
```