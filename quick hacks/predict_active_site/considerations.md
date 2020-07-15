Most prediction tools seem rather old, and a lot of them use 3D data. Seems a bit dumb, when I have 3D data I might as well just do docking.

Most recent: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5741869/  
meta-classifier using all the others. Train/Test split not really clear.  

General safe mode: When I perform better on my hold-out set than other methods, I need not worry about whether they trained on it or not. So in case things work, no problems.

- Download experimental/manual from uniprot
- homology partition
- LSTM with classifier layer, nothing special needed
- Try on top of LM

Uniprot: All organisms.  
Manual assertion: 99,652 Seqs  
Experimental assertion:  2,402 Seqs  
Need to also construct negative set -> Proteins without catalytic activity, otherwise non-presence might just be missing data.  

Last seq-only method apparently: http://biomine.cs.vcu.edu/datasets/CRpred/CRpred.html
https://www.sciencedirect.com/science/article/abs/pii/S0022519318300390?via%3Dihub


### Data Dump
https://www.uniprot.org/uniprot/?query=annotation%3A(type%3Aact_site%20evidence%3Aexperimental)&columns=id%2Centry%20name%2Creviewed%2Corganism%2Clength%2Cfeature(ACTIVE%20SITE)%2Cprotein%20names%2Csequence&sort=score&format=tab


### Model training

Cannot do the pointer network trick, as I have multiple positions to tag. Need other measure to deal with the sparsity.  
- AUPRC for evaluation
- Weighted loss?
- Site tolerance +- 1