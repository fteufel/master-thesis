# Try self-training using BERT

## Data
We use manually curated Uniprot sequences (swissprot). Reason only being that this set is smaller.  
Full Uniprot would be infeasible to train on, or even to predict.

```
NOT yourlist:M20201130A94466D2655679D1FD8953E075198DA81A7D5DG AND reviewed:yes
```
All sequences that are not yet in the training set.

## How to correctly predict with cross-validation?
For each split, I should only add sequences that were predicted with the split.  
