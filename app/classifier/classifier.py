from torch import nn
import torch.optim as optim
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_curve,precision_recall_fscore_support
import seaborn as sns

from sklearn import tree
import tensorflow as tf
from tensorflow.keras.utils import plot_model # type: ignore
from tensorflow.keras import models, layers # type: ignore
from nltk.corpus import stopwords
from torch.utils.data import DataLoader, Dataset
import torch
from torch.optim import Adam
from typing import List, Tuple, Callable
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import re
from collections import defaultdict, Counter

# save final_vocab
import pickle 
with open('final_vocab.pkl', 'rb') as f:
    final_vocab = pickle.load(f)

# Define tokenizer
class Tokenizer:
  def __init__(self):
    pass
  def tokenizer(query):
      # Regular expression to capture URLs
      url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

      # Replace URLs with '<url>'
      query = re.sub(url_pattern, '<url>', query, flags=re.IGNORECASE)

      # Regular expression to capture words, single quotation marks, and other punctuation separately
      pattern = r"""
      \w+|                  # Match sequences of word characters
      ['"]|                 # Match single or double quotes individually
      [^\w\s'"]             # Match any single character that is not a word character, whitespace, or quote
      """

      # Use re.findall with the VERBOSE and IGNORECASE flags to allow whitespace and comments in the regex string
      tokens = re.findall(pattern, query, re.VERBOSE | re.IGNORECASE)

      # Normalize tokens to lowercase and replace digits with '<num>'
      normalized_tokens = ['<num>' if token.isdigit() else token.lower() for token in tokens]
      return normalized_tokens


# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        output = self.fc(h_n[-1, :, :])
        return output

def predict(raw_text, pad_length=100):
    # Instantiate the model, define loss function, and optimizer
    token_size = 16214 # Adjust based on your vocabulary size
    embedding_dim = 64  # Adjust based on your preference
    hidden_size = 128
    output_size = 2  # Number of classes

    model = LSTMClassifier(token_size, embedding_dim, hidden_size, output_size)
    state_dict = torch.load("model_checkpoint_24.pth")
    # Apply the state dictionary to your model instance
    model.load_state_dict(state_dict)
    # device = 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    # tokenize data
    tokenizer = Tokenizer.tokenizer
    # Tokenize and convert to indices
    indexed_text = [final_vocab.get(token, final_vocab['<unk>']) for token in tokenizer(raw_text)]
    # Pad the query
    if len(indexed_text) < pad_length:
        padded_text = indexed_text + [final_vocab['<pad>']] * (pad_length - len(indexed_text))
    else:
        padded_text = indexed_text[:pad_length]

    output = model(torch.tensor(padded_text).to(device).unsqueeze(0))
    _, predicted = torch.max(output.data, 1)

    return predicted.item()