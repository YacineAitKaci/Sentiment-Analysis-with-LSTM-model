# Sentiment Analysis using LSTM model

## Project Overview
This project focuses on creating an LSTM model for sentiments classification using a custom dataset. The goal is to create a model that is capable of detecting the correct expressed emotion in a given sentence.

## Dataset 
The dataset used in this project comes from Kaggle. It consists of a collection of tweets expressing five different emotions, designed for sentiment analysis tasks. The dataset contains 20,000 rows, divided into three parts (training, test, and validation), with each row representing a sentence annotated with a sentiment class: Sadness (0) Joy (1) Love (2)Anger (3) Fear (4)
**Descriptive Statistics:**
Average sentence length: 19.14 words
Minimum sentence length: 2 words
Maximum sentence length: 66 words
**Class Distribution:**
Joy (1): 6,761 (33.80%)
Sadness (0): 5,797 (28.98%)
Anger (3): 2,709 (13.54%)
Fear (4): 2,373 (11.87%)
Love (2): 1,641 (8.21%)
Unknown label (5): 719 (3.60%)

## Libraries Used
- **TensorFlow/PyTorch:** Training deep learning model
- **Transformers (Hugging Face):** Pre-trained model and tokenizer loading
- **NLTK:** NLP utilities and tokenization
- **Gensim:** Pre-embeddings loading
- **NumPy & pandas:** Data handling and manipulation
- **Sklearn.metrics:** Evaluating model performance using diffrent metrics
- **Matplotlib** Visualization of training performance

## Installation
```bash
# Clone the repository
git clone https://github.com/YacineAitKaci/Sentiment-Analysis-with-LSTM-model.git

# Navigate to the project directory
cd Sentiment-Analysis-with-LSTM-model

# Install required libraries
pip install numpy pandas nltk gensim transformers torch tensorflow scikit-learn matplotlib
```

## How to Use
1. **Prepare the dataset:** Ensure the path to your 'train, test, validation' CSV is set correctly in the code 
2. **Run the training notebook:**
```bash
jupyter notebook NER_Fine_Tuning.ipynb
```
3. **Evaluate the model:** Inspect precision, recall, and F1-score to assess performance using diffrent pre-embedding.
4. **Run inference:** Use the created model for predicting sentiments on new text data.

...

Made with ❤️ for NLP & Machine Learning enthusiasts!