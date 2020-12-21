# Overview of results files


| file/directory | description |
|----------------|-------------|
|crossval_bert | Bert with standard global average loss
|crossval_compare_random_inits | Same as crossval_bert, repeated training with random starts |
|silascrf | Bert with Silas' CRF version, using default signalp dataset, kingdom-averaging MCCs |
|longer_context| Kingdom-averaged Silas CRF partition 0 with 128aa training sequences |
| brokengp \*  | signalp5/6 models that were trained on partitions where graphpart didnt work properly |
|signalp_5_model| retrained signalp5 model architecture results |
|signalp_6_model| bert-based model, results and plots |
|benchmark_performances_signalp5_paper | reproduction of benchmark tables of signalp5, to test for constistency |
|benchmark_performances_recomputed | benchmark tables with bert model, updated class memberships and removed seqs |
