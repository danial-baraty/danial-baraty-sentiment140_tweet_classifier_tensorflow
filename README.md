# Sentiment140 Tweet Classification with TensorFlow
A machine learning project applying a simple neural network in TensorFlow to classify tweets from the Sentiment140 dataset as positive or negative.

## Goal
Build and evaluate a text classification model to predict sentiment based on tweet content, using a tokenized and padded representation of tweets.

## Dataset
- Source: [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)  
- Description: Contains 1.6 million tweets labeled as positive or negative. For this project, a **balanced subset of 50,000 tweets** was used and preprocessed.

## Approach
- Preprocessed tweets: lowercasing, removing URLs, mentions, punctuation, and extra whitespace.
- Converted text to sequences using Keras Tokenizer and padded them to fixed length.
- Built a simple TensorFlow neural network with Embedding, GlobalAveragePooling, and Dense layers.
- Evaluated performance with accuracy on a train/test split.

## How to Use
- Clone this repo: `git clone https://github.com/danial-baraty/danial-baraty-sentiment140_tweet_classifier_tensorflow.git`
- Install dependencies: `pip install -r requirements.txt`
- Run the Jupyter notebook locally or via Colab.
