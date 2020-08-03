# Data downloading and preprocessing

# 04/06/2020

- Download data from UniProt
- Homology partition and split in train, val, test
- Convert to LMDB database files


   """
    The high quality dataset is homology partitioned into training, validation, and test set. 
    We employ 60% for training, 10% for validation and 30% for testing. 
    When partitioning we first cluster the proteins based on a similarity threshold of 20% identity. 
    This means that proteins with an identity higher than this threshold are grouped in the same cluster. 
    For the clustering task, we utilize the MMSeqs2 tool (Steinegger and Söding, 2017). 
    Each cluster is then assigned to one of the three possible partitions, taking into consideration that the proportions of taxonomical domains, 
    and fragmented proteins, are the same across all partitions.
    """

- run download_data with target dir as arg
- run mmseqs_clustering_pipeline with both exp files as args to cluster at once
- reformat .fasta to tsv and clean duplicates
- join the cluster identifier to the .tsv's maybe rather do in python

- align other two datasets to ref clustering

# 05/06/2020

- Model architecture: I guess in the interest of training time I'll use AWD-LSTM. 
- Baseline model: What would you recommend to use as the baseline, to compare whether a specific model brings improvement?
Eukaryotes only, or all domains?
- Plasmodium training data:

I thought about working along the Unilanguage pipeline:
observed full : 1066    seqs
observed frag : 4932    seqs
predicted full: 351440  seqs
predicted frag: 57499   seqs
If I got it right, the Unilanguage pipeline would require me to cluster all the observed, then tr/test/val split in a "cluster size weighted" way.
Obviously the observed seq count is very low for plasmodium only. Clustering doesn't work very well

'taxonomy:Plasmodium [5820] AND (existence:"Evidence at protein level [1]" OR existence:"Evidence at transcript level [2]")'
'taxonomy:Plasmodium [5820] AND (existence:"Predicted [4]" OR existence:"Inferred from homology [3]")'


#### How to do train_test_val split on baseline data
## Experimental outline 
#### Dataset

- Homology partition
- Now I need to make sure that the baseline training set does not contain the plasmodium test set.
- A) perform homology partitioning on Uniref without Plasmodium, then add the Plasmodium splits to the Uniref splits
- B) Align complete Uniref to Plasmodium clusters

#### Modeling
- Use AWD-LSTM with SGD
- same model, same hyperparams, for both datasets
- TO DECIDE: different or same validation sets?
- Intuitively using plasmodium only on full data seems absurd, but also unfair if I do not do it


# 15/06/2020

- Baseline: All Eukarya
- Discard fragments, they don't contain signal peptides (model would still interpret as beginning of seq, although is at actually at random position in a seq)
- Use AWD-LSTM or SHA-LSTM

#### Dataset
- Query UniProt for Eukarya, also take taxon info
- Cluster with mmseqs2
- Split train test val, ensuring equal distribution of plasmodium proteins
- Get dataset summary statistics, length aa usage etc

# 16/06/2020

#### Notes to AWD-LSTM implementation
- training process seems counterintuitive for proteins. Salesforce implementation treats whole training data as one long sequence, and pulls 'batches' of a specified length from this seq. Final hidden state is start hidden state of next 'batch'
- Makes sense if my training data is a long corpus of sentences that possibly form longer texts. For proteins, there is no good reason to keep the hidden state
- learning rate is also scaled by sequence length. In a padded minibatch this does not work.
- in a minibatch i kind of emulate the variable length if i exclude pad tokens from the loss calculation.

- TAPE has everything i need, only need to put activation regularization inside the model, where the loss is
- Decreasing LR also needs to be added

TODO
- clean up TAPE AWD-LSTM implementation
- add activation regularization
- make new train.py with decreasing LR
- interface to hyperparam opt, maybe even via CLI


# 17/06/2020


#### AWDLSTM explanation from Jose
The way that the AWD-LSTM performs the computation is called Truncated BPTT. Let me give you an example with proteins. Lets say that we have 4 proteins:
MRARRL
MGARRLS
MLLARRR
MERRAL
 
The way that this works is that the proteins are “concatenated” with an end of sequence character, lets call it ‘>’, and divided in minibatches, lets say we want a max length of 3 to BPTT and 2 minibatches:
MRARRL>MGARRLSS>MLARRR>MERRAL>
Max len of 3:
MRA | RRL | >MG | ARR | LSS | >ML | ARR | R>M | ERR | AL>
2 minibatches
MRA | RRL | >MG | ARR | LS>
MLL | ARR | R>M | ERR | AL>
 
This means that when doing the BPTT it only backpropagates 3 aa but the hidden state of is passed to the next batch. So the last hidden state of processing the batch:
MRA
MLL
will be transferred as the initial hidden state of the next batch:
RRL
ARR

training corpus: 
MRA RRL >MG ARR LSS >ML ARR R>M ERR AL>
MRA RRL >MG ARR LS> MLL ARR R>M ERR AL>


#### Overview of changes made/to be made to TAPE
tape/models/ added AWDLSTM
tape/datasets tasks 'eukarya_language_modeling' 'plasmodium_language_modeling
training.py added decreasing LR
main.py added CLI arg for decreasing LR (actually better just write own training thing)


#### Implemented
- AWD-LSTM in TAPE setup
- train_awdlstm_lm.py : custom training script to support bptt training strategy

_README to add to AWD_LSTM experiment directory_

- Somewhat clean and structured AWD-LSTM implementation
- Inherits base classes from TAPE (pip install tape-proteins)
- train_awdlstm_lm implements a bptt training pipeline
- training data is expected to be a directory with train.txt, valid.txt and test.txt. Plain text, read line-by-line. End of line gets the end of sequence token.
- Then everything is concatenated, this is cut into batch_size strings. These batch_size strings are processed piecewise with length bptt


- supports resetting the hiddens state after seeing an eos token
for this, LSTMcell gets token sequence as input together with the embeddings
at each step, tokens from previous step are given. if previous step token is reset token, reset hidden for this observation to 0 before updating
to not have this behaviour, leave reset_token_id at default (negative, won't be in any tokenizer probably) or pass None as tokens



### Hyperparameter optimization:
Weights & Biases:
- export WANDB_PROJECT is required.
- Parameter sweep works by starting a daemon script that calls train.py with hyperparameters as cl arguments.
- Hyperparameter names need to match argparse arguments
- **vars(args) unpacks args so that they can be fed to the AWDLSTMConfig constructor.



## Data too big to load into memory.

#make one line on commandline
sed -z 's/\n/<sep>/g'

a: get number of characters wc -m
b: get number of eos tokens: grep -o '<sep>' | wc -l

total token length: a- 2*b

--> now i have a one line datafile. what next. dont know

divide total token length by batch_size -> len_batch

    len_batch = total_token_length // batch_size
    batch indices: [0: len_batch], [len_batch:len_batch+1] ... [:len_batch+n]


### Solution: Virtual indexing, implemented in hd5 Dataloader
- data is tokenized and concatenated, saved as hdf5 (takes 100gb ram, but only needs to be done once). 1D array of length total_seq_len.
- calculate tokens_per_batch from total_seq_len/batch_size.
- specify offsets for the batch_size batches into the 1D array. batch i starts at tokens_per_batch*i (batch enumeration starting at 0).
- compute bptt lenghts as before, starting at 0
- when accessing an element, add offsets to start:end, and build indexing matrix. This yields a (bptt,batch_size) array, each batch starting at the correct position. 


# 29/06/2020

### Status
- Models built.
- Dataloader with virtual indexing and buffering
- wandb hyperparameter tuning

### Fine-tuning models
from https://www.biorxiv.org/content/10.1101/2020.01.23.917682v1.full
_The sequence set was filtered for length (kept all <500 amino acids) and Levenshtein distance from avGFP (kept all <400), and sequences with non-standard amino acids were removed, yielding 79,482 sequences. We selected a 10% “out of distribution set” by sampling each sequence with a probability proportional to the 4th power of the edit distance. A 10% in-distribution set was selected uniformly randomly. We initialized the weights of the 1900 dimensional UniRep mLSTM with the globally pre-tained weights and trained for 13,500 iterations with early stopping68, 69, until the outer validation set loss began to increase. This model was used to produce the representations for eUniRep 2 as named above._

__This means for us__: Run standard learning setup with plasmodium datasets. Evaluate plasmodium validation set performance and stop early.   
Possible Considerations:
 - Triangular learning rate




 # 30/06/2020

 ## Progress Meeting
 - Use as many GPUs as you can get
 - 6000 steps, then update lr if not better
 - Validation pass: Find out how long it takes
 - prepare n-gram baseline
 - Evaluate UniRep perplexity on Plasmodium/Eukarya sets
 - Read on specific LMs for text categories


 # 06/07/2020

 ## CRF Signal peptide detection
  - Using stop token? Then also need stop label in CRF.   
  Now, just not added when tokenizing padded batches.
  - Fix mask, can be provided as arg to model. No need to create in CRF.


  # 07/07/2020

  ## Low perplexity of plasmodium model

  - Try homology reduction. MMSEQs2 output also gives representatitve sequences. These are my homology-reduced dataset.e



# 03/08/2020

## Perplexity experiment completed
 - finetuned models are better.
 - specific models is good too.

--> Need a downstream task now. Predict SPs.

- Plasmodium paper: Train on experimental sequences only (no plasmodium!), evaluate on non-experimental plasmodium sequences.
- Use SignalP training data
- Try classification layer without CRF first.