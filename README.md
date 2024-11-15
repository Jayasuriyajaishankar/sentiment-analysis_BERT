# sentiment-analysis_BERT

Flipkart Reviews Sentiment Analysis
This project performs sentiment analysis on Flipkart product reviews using a fine-tuned BERT model. Reviews are classified as either positive or negative based on user ratings. The project uses transformers and torch libraries for model implementation.

Table of Contents
Project Overview
Dataset
Data Preprocessing
Model Training
Evaluation
Prediction
Results
Setup and Installation
Usage

Project Overview:
The project is designed to classify user reviews as positive or negative based on the rating and review content. A BERT model (bert-base-uncased) is fine-tuned on this labeled data, with reviews rated 3 or above as positive and those rated 2 or below as negative.

Dataset:
This project uses the Flipkart Product Reviews dataset. Please download the dataset from https://www.kaggle.com/datasets/naushads/flipkart-reviews and place it in the working directory with the name flipkart_reviews_dataset.csv.

Data Preprocessing:
Dataset Structure: The dataset contains the following columns:
product_id, product_title, rating, summary, review, location, date, upvotes, downvotes
Labeling: A label column is created:
Ratings >= 3 are labeled as 1 (Positive)
Ratings <= 2 are labeled as 0 (Negative)
Tokenization: BERTTokenizer is used to tokenize the reviews with padding and truncation to a max length of 128 tokens.

Model Training:
Model: BERTForSequenceClassification is initialized for binary classification.
Optimization: AdamW optimizer with a learning rate of 2e-5.
DataLoader: TensorDataset and DataLoader split the dataset into training and validation sets with batch size 16.
Training Loop: The model is trained for 3 epochs with gradient clipping and a linear learning rate scheduler.

Evaluation:
After training, the model is evaluated on the validation set:
Accuracy: Overall accuracy on validation data is calculated.
Loss: Validation loss is computed.
Confusion Matrix: A confusion matrix visualizes the model’s performance.
Prediction
A custom function predict_sentiment(review_text) predicts sentiment for individual reviews. The function:

Tokenizes input text
Generates prediction by passing tokenized data through the model
Returns 1 for positive sentiment and 0 for negative sentiment.

Results:
Validation Accuracy: 96.87%
Validation Loss: 0.134
Confusion Matrix: The confusion matrix is plotted using seaborn for detailed performance visualization.

Setup and Installation:
1)Environment: Python 3.7 or later
2)Install Requirements
pip install pandas transformers torch sklearn matplotlib seaborn

Usage:
Training: Run the training loop to fine-tune the BERT model.
Evaluation: Evaluate model performance on the validation set.
Prediction: Use predict_sentiment() function for individual review predictions.

Acknowledgments:
This project uses the Hugging Face transformers library for NLP tasks and torch for model training and optimization. Special thanks to the open-source contributors of these libraries.






