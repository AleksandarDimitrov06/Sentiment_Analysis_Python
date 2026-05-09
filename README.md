# Sentiment Analysis Model

This project features a simple Neural Network built with PyTorch that performs binary sentiment analysis on text data. It reads training and testing data from TSV files, converts the text into numerical features using TF-IDF vectorization (excluding custom stop words), and trains a feed-forward neural network to classify the sentiment as either positive or negative. 

Once training and evaluation are complete, the script launches an interactive prompt where you can type in your own reviews to test the model's predictions in real-time.

## Prerequisites

Before starting, ensure you have Python installed, and then install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Start the Project

1. Ensure the dataset files (`sentiment_data.tsv` and `sentiment_test_data.tsv`) are in the project's root directory.
2. Run the main script from your terminal:

```bash
python Sentiment_Analysis.py
```

3. The script will automatically output GPU availability, load the data, train the model, and print the test accuracy.
4. After evaluation, an interactive prompt will appear. Type any sentence to see the model's sentiment prediction (Positive 🟢 or Negative 🔴) and its confidence score. Type `Close` to exit the program.
