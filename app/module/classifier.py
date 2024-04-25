import torch
import re
from torch import nn
import warnings
warnings.filterwarnings('ignore')


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


# Define the LSTM model based WAF
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


# load final_vocab
import os
import pickle
current_script_path = os.path.dirname(__file__)
vocab_file_path = os.path.join(current_script_path, 'final_vocab.pkl')
with open(vocab_file_path, 'rb') as f:
    final_vocab = pickle.load(f)
# with open('final_vocab.pkl', 'rb') as f:
#      final_vocab = pickle.load(f)

# LSTM model based WAF
class WAF:
    def __init__(self):
        self.model = LSTMClassifier(len(final_vocab), 64, 128, 2)
        self.model.load_state_dict(torch.load(os.path.join(current_script_path, 'model_checkpoint_24.pth'), map_location=torch.device('cpu')))
        # self.model.load_state_dict(torch.load('model_checkpoint_24.pth', map_location=torch.device('cpu')))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, raw_text, pad_length=100):
        # tokenize data
        tokenizer = Tokenizer.tokenizer
        # Tokenize and convert to indices
        indexed_text = [final_vocab.get(token, final_vocab['<unk>']) for token in tokenizer(raw_text)]
        # Pad the query
        if len(indexed_text) < pad_length:
            padded_text = indexed_text + [final_vocab['<pad>']] * (pad_length - len(indexed_text))
        else:
            padded_text = indexed_text[:pad_length]

        output = self.model(torch.tensor(padded_text).to(self.device).unsqueeze(0))
        _, predicted = torch.max(output.data, 1)

        return predicted.item()
    

if __name__ == '__main__':
    waf = WAF()
    ret = waf.predict('hello')
    print(ret)