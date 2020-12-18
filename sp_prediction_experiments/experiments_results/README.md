# Overview of results files


| file/directory | description |
|----------------|-------------|---
|crossval_bert | Bert with standard global average loss
|crossval_compare_random_inits | Same as crossval_bert, repeated training with random starts |
|silascrf | Bert with Silas' CRF version, using default signalp dataset, kingdom-averaging MCCs |
|longer_context| Kingdom-averaged Silas CRF partition 0 with 128aa training sequences
| brokengp \*  | signalp5/6 models that were trained on partitions where graphpart didnt work properly
