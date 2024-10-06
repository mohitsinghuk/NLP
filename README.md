# NLP

## Overview
This project is focused on evaluating the performance of different Natural Language Processing (NLP) models for a multi-class text classification problem. The goal is to classify textual data, particularly abbreviations in scientific documents, using models such as Recurrent Neural Networks (RNNs) with Bidirectional Long Short-Term Memory (LSTM) and ALBERT, as well as Support Vector Machines (SVM) with word embeddings like GloVe and FastText. This document summarizes the analysis, model setups, experimentation, and the best practices identified during the testing.


## Project Structure

Introduction: Overview of multi-class text classification and the significance of accurate textual data interpretation in NLP.

Dataset Analysis: Exploration of the PLOD-CW dataset designed for abbreviation detection, including token frequency, NER distribution, and visualization using word clouds and heatmaps.

Experimental Setups: Details the four main experimental approaches, including RNN with LSTM & ALBERT, and SVM with GloVe & FastText embeddings.

Model Testing and Evaluation: Evaluation of the models using precision, recall, F1-score, and accuracy to determine the best-performing model.

Best Model and Recommendations: Identifies the most effective model and suggests potential improvements for further optimization.

Conclusion & Future Improvements: Summary of overall model performance and proposed strategies for enhancement.

References: Resources and literature that informed the project.


## Dataset Description

PLOD-CW Dataset: Designed for NLP tasks such as abbreviation detection within scientific documents. Each data point contains:

Tokens extracted from text.

Part-of-Speech (POS) and Named Entity Recognition (NER) tags.


## Data Analysis Visualizations:

Token Frequencies: Top 10 frequent tokens for training and test data.

NER Distribution: Pie charts for NER tag distribution in both training and test datasets.

Word Cloud: Highlights the most common words in the dataset.

Text Lengths: Analysis using cumulative distribution function (CDF) plots and heatmaps for text length variations.


## Model Implementations and Experimental Setups

1. RNN with Bidirectional LSTM and ALBERT

Why RNN & LSTM?: Effective for sequence labeling and contextual understanding due to the ability to maintain state over sequences.

ALBERT: A lightweight transformer model that reduces computational complexity while preserving model performance.

Training: Tokenization, padding, and conversion of tags to integer sequences were performed, followed by model training using cross-entropy loss.


2. SVM with GloVe and FastText Embeddings

SVM Overview: Effective in high-dimensional spaces and versatile with different kernel functions for linear and non-linear data separation.

GloVe Embeddings: Trained on global word co-occurrences to provide dense semantic vector representations.

FastText Embeddings: Captures subword information, handling rare or unknown words effectively.

Experimental Setup: Tokenization, embedding initialization, document vector creation, SVM setup, and evaluation using classification reports.


3. Loss Functions and Optimizers

CrossEntropyLoss: Used for classification tasks paired with softmax activation for better model performance.

KLDivLoss: Tested but found less effective than CrossEntropyLoss for this particular task.

AdamW Optimizer: A variant of the Adam optimizer that includes a weight decay correction for better convergence and generalization.


4. Hyperparameter Tuning

GridSearchCV: Applied for optimizing parameters like C, gamma, and kernel type for SVM.

Optimization Goals: Improve precision, recall, and F1-scores across different classes.


## Model Testing and Results

RNN + LSTM & ALBERT: Demonstrated high precision and recall for the B-LF class, indicating strong model performance.

SVM with GloVe & FastText: Moderate success with certain classes, though struggled to classify others effectively.

Loss Function Evaluation: CrossEntropyLoss with AdamW optimizer showed overall better performance.

Overall Accuracy: Models performed well on some classes but struggled with underrepresented classes, indicating a need for balanced datasets or further model tuning.


## Best Model Recommendation

RNN with Bidirectional LSTM and ALBERT is identified as the best model due to:

High performance on key metrics (precision, recall, and F1-score).

Ability to handle complex sequence data effectively.


## Suggestions for Improvement

Hyperparameter Tuning: Adjust learning rates, batch sizes, and dropout rates.

Model Architecture Enhancements: Implement additional LSTM layers or attention mechanisms for better context understanding.

Data Augmentation: Augment training data to improve model robustness.

Advanced Embeddings: Experiment with pre-trained embeddings or dimensionality adjustments.


## Conclusion & Future Work
The project successfully evaluated multiple NLP models for multi-class text classification. The findings highlight the strengths of RNNs with Bidirectional LSTMs in sequence labeling and suggest avenues for improving model performance through advanced preprocessing and hyperparameter optimization.


## Getting Started
Prerequisites
Python 3.x

Libraries: TensorFlow, Keras, scikit-learn, Gensim, Matplotlib, Seaborn

How to Run the Project

Clone the Repository: Download the project files.

Install Dependencies: Run pip install -r requirements.txt.

Prepare Data: Load and preprocess the dataset (PLOD-CW).

Run Experiments: Execute model scripts in Jupyter Notebooks for different experiments.

Evaluate Models: Use provided evaluation scripts to assess model performance.
