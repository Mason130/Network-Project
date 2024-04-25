import re
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

url1 = './Modified_SQL_Dataset.csv'
url2 = './SQLiV3.csv'
url3 = './sqli.csv'
url4 = './sqliv2.csv'
df1 = pd.read_csv(url1, header=None, skiprows=1)
df2 = pd.read_csv(url2, header=None, skiprows=1)
df3 = pd.read_csv(url3, header=None, skiprows=1, encoding='utf-16')
df4 = pd.read_csv(url4, header=None, skiprows=1, encoding='utf-16')

# deal with df2
print(len(df2))
# Filter the DataFrame to keep only rows where the second column has '0' or '1'
df2 = df2[df2.iloc[:, 1].isin(['0', '1'])].iloc[:, [0,1]]
df2.iloc[:,1] = df2.iloc[:,1].astype(int)
print(len(df2))

# Select first two columns
df1_selected = df1.iloc[:, :2]
df2_selected = df2.iloc[:, :2]
df3_selected = df3.iloc[:, :2]
df4_selected = df4.iloc[:, :2]
# Concatenate the selected columns
df = pd.concat([df1_selected, df2_selected, df3_selected, df4_selected], axis=0)
# df = df1_selected
df = df.rename(columns={0: 'query', 1: 'label'})
df = df.dropna()
df = df.reset_index(drop=True)
# Show the concatenated DataFrame
print(len(df))
df.head()

'''Prepare training and testing data'''
X_txt = list(df['query'])
y = list(df['label'])
# print ratio of positive and negative
print(y.count(0) / len(y))
print(y.count(1) / len(y))
# Train test split
X_train_txt, X_test_txt, y_train, y_test = train_test_split(X_txt, y, test_size = 0.1)

'''Tokenizer'''
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
  

# Tokenize and record vocabulary
tokenizer = Tokenizer.tokenizer
vocab_counter = Counter()
for query in X_train_txt: # [0:10000]
    # print(query)
    tokens = tokenizer(query)
    vocab_counter.update(tokens)
# Print all unique vocabularies
# print(vocab_counter)
print('total vocab length', len(vocab_counter))

## Ignore all tokens appearing in less than 1% of data
# Calculate the 5% threshold
threshold = len(X_train_txt) * (2/len(X_train_txt))
# Filter vocabularies by occurrence
filtered_vocab = {token: count for token, count in vocab_counter.items() if count > threshold}
# Handling tokens that appear less frequently than the threshold
final_vocab = {token: i+1 for i, token in enumerate(filtered_vocab)}  # Start indexing from 1
final_vocab['<unk>'] = 0  # Unknown tokens are indexed as 0
final_vocab['<pad>'] = len(final_vocab)  # Padding tokens are indexed as last index
# save final_vocab
import pickle
with open('final_vocab.pkl', 'wb') as f:
    pickle.dump(final_vocab, f)
with open('final_vocab.pkl', 'rb') as f:
    final_vocab = pickle.load(f)
# print(final_vocab)
print('reduced vocab length', len(final_vocab))

'''Tokenize train data'''
temp_collector = []
X_train_encoded = []
pad_length = 100 #max(len(query) for query in temp_collector)
for query in X_train_txt:
    # Tokenize and convert to indices
    indexed_query = [final_vocab.get(token, final_vocab['<unk>']) for token in tokenizer(query)]
    # Pad the query
    if len(indexed_query) < pad_length:
      padded_query = indexed_query + [final_vocab['<pad>']] * (pad_length - len(indexed_query))
    else:
      padded_query = indexed_query[:pad_length]
    # Append the padded query to the list
    X_train_encoded.append(padded_query[:pad_length])  # Ensure it does not exceed pad length

'''Tokenize test data'''
temp_collector = []
X_test_encoded = []
for query in X_test_txt:
    # Tokenize and convert to indices
    indexed_query = [final_vocab.get(token, final_vocab['<unk>']) for token in tokenizer(query)]
    # Pad the query
    if len(indexed_query) < pad_length:
      padded_query = indexed_query + [final_vocab['<pad>']] * (pad_length - len(indexed_query))
    else:
      padded_query = indexed_query[:pad_length]
    # Append the padded query to the list
    X_test_encoded.append(padded_query[:pad_length])  # Ensure it does not exceed pad length

'''Custom Dataset'''
# Define a simple dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.int64)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': self.labels[idx]}

# Instantiate the dataset and dataloaders
train_dataset = TextDataset(X_train_encoded, y_train)
test_dataset = TextDataset(X_test_encoded, y_test)
# Assuming train_dataset is already created
total_train_samples = len(train_dataset)
train_size = int(0.8 * total_train_samples)
val_size = total_train_samples - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
batch_size = 24
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

'''Model Defining and Training'''
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


# Instantiate the model, define loss function, and optimizer
token_size = len(final_vocab) # Adjust based on your vocabulary size
embedding_dim = 64  # Adjust based on your preference
hidden_size = 128
output_size = 2  # Number of classes
model = LSTMClassifier(token_size, embedding_dim, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)
model.to(device)

'''Training loop'''
epochs = 25
train_losses = []
val_losses = []
train_accuracy_list = []
val_accuracy_list = []

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_correct = 0
    total_samples = 0
    running_loss = 0
    for batch in train_loader:
        texts, labels = batch['text'], batch['label']
        texts, labels = texts.to(device), labels.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(texts)
        # Compute loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    train_accuracy = total_correct / total_samples
    train_losses.append(running_loss / len(train_loader))
    train_accuracy_list.append(train_accuracy)
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            texts, labels = batch['text'], batch['label']
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            # Compute validation loss
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # Save the model checkpoint
    torch.save(model.state_dict(), f'model_checkpoint_{epoch}.pth')
    # Average validation loss and accuracy
    val_losses.append(val_loss / len(val_loader))
    val_accuracy = correct / total
    val_accuracy_list.append(val_accuracy)
    # Print epoch summary
    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_losses[-1]:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
'''model validation'''
model.eval()
with torch.no_grad():
    for batch in next(iter(test_loader)):
        text, label = batch['text'], batch['label']
        text, label = text.to(device), label.to(device)
        outputs = model(text)
        _, predicted = torch.max(outputs.data, 1)
        print(text)
        print(f'Predicted: {predicted}, Actual: {label}')

idx = 1232
output = model(torch.tensor(X_test_encoded[idx]).to(device).unsqueeze(0))
_, predicted = torch.max(output.data, 1)
print(predicted)
print(" ")
print(X_test_txt[idx], y_test[idx])
print(X_test_encoded[idx])

raw_text = "name = ' OR 'a'='a';--"

idx = 73812
raw_text = df.iloc[idx,0]
raw_text = "name = ' OR 'a'='a';-- and password = any"
# Tokenize and convert to indices
indexed_text = [final_vocab.get(token, final_vocab['<unk>']) for token in tokenizer(raw_text)]
# Pad the query
if len(indexed_text) < pad_length:
  padded_text = indexed_text + [final_vocab['<pad>']] * (pad_length - len(indexed_text))
else:
  padded_text = indexed_text[:pad_length]
output = model(torch.tensor(padded_text).to(device).unsqueeze(0))
_, predicted = torch.max(output.data, 1)
print(raw_text, 'LABEL: ', df.iloc[idx,1])
print(predicted)
