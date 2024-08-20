# Academic Project: Classification-of-Sentiment-in-reviews-from-Yelp-Customers

This repository provides a Python script for fine-tuning and comparing large language models on sentiment analysis tasks. It includes implementations for both a baseline model (BERT) and an advanced model (RoBERTa), along with a comparison of their performance against state-of-the-art (SOTA) results.

### Key Features:
#### Model Implementations:

BERT: Fine-tunes the bert-base-uncased model for sentiment analysis.
RoBERTa: Fine-tunes the roberta-base model, providing a potentially more advanced approach.

#### Data Handling:

Uses the first 1000 reviews from the yelp_review_full dataset for both training and evaluation.
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

## Analysis of Results:

The results indicate that both the fine-tuned BERT and RoBERTa models underperform significantly compared to the state-of-the-art (SOTA) accuracy of 0.85. Specifically:

BERT Accuracy: 0.42
RoBERTa Accuracy: 0.54
This suggests that while RoBERTa outperformed BERT in this particular case, both models are still far from achieving SOTA performance. Here are a few suggestions to improve the models:

### Possible Improvements:
#### - Increase Data Size:

Expanding the dataset beyond the first 1,000 reviews could provide the models with more diverse examples, potentially leading to better generalization.

#### - Use Larger Models:

Although you used smaller models to reduce resource consumption, switching to larger versions like bert-base-uncased or roberta-base could improve accuracy, given sufficient computational resources.

#### - Hyperparameter Tuning:

Experiment with different learning rates, batch sizes, and number of epochs.
Use techniques like learning rate scheduling or dynamic batch sizing.

#### - Data Augmentation:

Applying techniques like back-translation or synonym replacement to artificially increase the size and diversity of the training dataset.

#### - Domain-Specific Pretraining:

If possible, pretrain the model on a large corpus of Yelp reviews or similar sentiment analysis data before fine-tuning on the labeled data.

#### - Ensemble Methods:

Combining the predictions of BERT and RoBERTa through ensembling techniques might improve overall performance.

#### - More Complex Architectures:

Exploring hybrid models or integrating attention mechanisms tailored to the specific nature of sentiment analysis in reviews could lead to better results.
