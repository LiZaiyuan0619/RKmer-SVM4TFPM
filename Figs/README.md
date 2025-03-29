# README

## Supplementary Figures

We fine-tuned the ProtBERT model for 20 epochs on a single NVIDIA A40 GPU with a batch size of 1. The corresponding data were directly retrieved from the Weights & Biases (wandb) logging system. The recorded metrics include MCC, ACC, AUC, F1 score, Loss, Sensitivity (SN), Specificity (SP), as well as GPU memory usage.

## Computational Efficiency

**(a)** We randomly selected 1,000 sequences with an average length of 539.86 amino acids (ranging from 51 to 4,834) to evaluate inference performance. The total processing time was 82.84 seconds, averaging approximately 0.083 seconds per sequence (with individual sequence processing times ranging from 0.0816 to 0.0884 seconds). This translates to a processing rate of roughly 12 sequences per second, which is highly practical for large-scale genomic analysis. The computational complexity scales linearly with the number of sequences, though there is some variation based on sequence length.

**(b)** During the ProtBERT fine-tuning process with parameters batch_size=1 and learn_rate=1e-5, the GPU memory usage was approximately 13GB. The complete training process over 20 epochs required nearly 1 hour, with each epoch taking approximately 3 minutes. As detailed in the supplementary materials, performance metrics stabilized relatively quickly (around epoch 3), with fluctuations not exceeding 3% thereafter.