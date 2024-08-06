# Academic Project: Classification-of-Sentiment-in-reviews-from-Yelp-Customers
This repository contains a Python script for fine-tuning the BERT model for sentiment analysis using the transformers and datasets libraries from Hugging Face. The code demonstrates the end-to-end process of adapting a pre-trained BERT model for classifying sentiment in customer reviews.

### Key Features:
Model and Tokenizer: Utilizes bert-base-uncased as the pre-trained model and tokenizer.
Dataset: Uses the yelp_review_full dataset, which is preprocessed to fit the model’s input requirements.
Fine-Tuning: Configures training parameters such as learning rate, batch size, and number of epochs, and uses the Trainer class for the training loop.
Evaluation: Evaluates model performance using accuracy as the metric, with results printed after training.

### Usage:

Install Dependencies: Ensure you have the necessary libraries installed:
###### pip install transformers datasets torch

### Run the Script: Execute the script to fine-tune the BERT model on the sentiment analysis task. The model and results will be saved in the ./results directory.

### Evaluate: After training, the script evaluates the model on the test set and prints the accuracy results.

This script provides a straightforward example of how to fine-tune a large language model for a specific NLP task and is suitable for anyone looking to adapt pre-trained models for custom applications.
