# Academic Project: Classification-of-Sentiment-in-reviews-from-Yelp-Customers

This repository provides a Python script for fine-tuning and comparing large language models on sentiment analysis tasks. It includes implementations for both a baseline model (BERT) and an advanced model (RoBERTa), along with a comparison of their performance against state-of-the-art (SOTA) results.

### Key Features:
#### Model Implementations:

BERT: Fine-tunes the bert-base-uncased model for sentiment analysis.
RoBERTa: Fine-tunes the roberta-base model, providing a potentially more advanced approach.

#### Data Handling:

Uses the first 10,000 reviews from the yelp_review_full dataset for both training and evaluation.
Training and Evaluation:

Configures training arguments including learning rate, batch size, number of epochs, and evaluation strategy.
Implements early stopping and model checkpointing to optimize training.
Comparison with SOTA:

Compares the performance of the fine-tuned models against predefined SOTA benchmarks.
Outputs accuracy metrics and assesses whether the custom RoBERTa model outperforms existing methods.

#### Run the Script:
Execute the script to fine-tune both BERT and RoBERTa models, and evaluate their performance.

#### View Results:
The script provides a comparison of the fine-tuned models' accuracy against SOTA results, helping to assess the effectiveness of the custom model improvements.

This code demonstrates how to adapt and evaluate different transformer models for sentiment analysis, offering insights into achieving better performance than traditional SOTA methods.
