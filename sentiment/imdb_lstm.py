"""
LSTM model for sentiment analysis on IMDB dataset
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

import numpy as np
import pandas as pd
from string import punctuation
from collections import Counter

from sklearn import metrics
from sklearn.metrics import classification_report




class SentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(
        self,
        vocab_size,
        output_size,
        embedding_dim,
        hidden_dim,
        n_layers,
        drop_prob=0.5,
    ):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True
        )

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
            )
        else:
            hidden = (
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
            )

        return hidden



def pad_features(reviews_int, seq_length):
    """
    Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    """
    features = np.zeros((len(reviews_int), seq_length), dtype=int)

    for i, review in enumerate(reviews_int):
        review_len = len(review)

        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length - review_len))
            new = zeroes + review
        elif review_len > seq_length:
            new = review[0:seq_length]

        features[i, :] = np.array(new)

    return features




# Prepare the dataset
df_train = pd.read_json(path_or_buf="data/imdb_prepared_train.jsonl", lines=True)
n_train = len(df_train) // 50 * 50
df_train = df_train.iloc[:n_train]

df_test = pd.read_json(path_or_buf="data/imdb_prepared_valid.jsonl", lines=True)
df = pd.concat([df_train, df_test], ignore_index=True)
df.columns = ["review", "sentiment"]

## Preprocessing
df.sentiment = df.sentiment.apply(lambda x: "positive" if x == 1 else "negative")

df["review"] = df["review"].apply(lambda x: x.lower())
df["clean_text"] = df["review"].apply(
    lambda x: "".join([c for c in x if c not in punctuation])
)

df["len_review"] = df["clean_text"].apply(lambda x: len(x))
all_text2 = df["clean_text"].tolist()

# Create Vocab to Int mapping dictionary

all_text2 = " ".join(all_text2)

words = all_text2.split()
count_words = Counter(words)

total_words = len(words)
sorted_words = count_words.most_common(total_words)

vocab_to_int = {w: i for i, (w, c) in enumerate(sorted_words)}
vocab_to_int = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}

reviews_split = df["clean_text"].tolist()

# Encode the words
reviews_int = []
for review in reviews_split:
    r = [vocab_to_int[w] for w in review.split()]
    reviews_int.append(r)


# Encode the labels
labels_split = df["sentiment"].tolist()
encoded_labels = [1 if label == "positive" else 0 for label in labels_split]
encoded_labels = np.array(encoded_labels)

reviews_len = [len(x) for x in reviews_int]
reviews_int = [reviews_int[i] for i, l in enumerate(reviews_len) if l > 0]
encoded_labels = [encoded_labels[i] for i, l in enumerate(reviews_len) if l > 0]


# Padding / Truncating the remaining data
features = pad_features(reviews_int, 200)
len_feat = len(features)

# Training, Validation, Test Dataset Split
## 45000, ~4000, 1000
train_x = features[0:45000]
train_y = encoded_labels[0:45000]
valid_x = features[45000:len(df_train)]
valid_y = encoded_labels[45000:len(df_train)]
test_x = features[len(df_train) :]
test_y = encoded_labels[len(df_train) :]


train_y = np.array(train_y)
test_y = np.array(test_y)
valid_y = np.array(valid_y)


# Dataloaders and Batching

## create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

## dataloaders
batch_size = 50
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


# Instantiate the network

vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2
net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)


train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print("Training on GPU.")
else:
    print("No GPU available, training on CPU.")


# Training Loop
lr = 0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

epochs = 4  
counter = 0
print_every = 100
clip = 5  # gradient clipping

if train_on_gpu:
    net.cuda()

net.train()
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    for inputs, labels in train_loader:
        counter += 1

        if train_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state
        h = tuple([each.data for each in h])

        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                val_h = tuple([each.data for each in val_h])

                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print(
                "Epoch: {}/{}...".format(e + 1, epochs),
                "Step: {}...".format(counter),
                "Loss: {:.6f}...".format(loss.item()),
                "Val Loss: {:.6f}".format(np.mean(val_losses)),
            )


# Testing

test_losses = []
num_correct = 0

true_list = []
pred_list = []
prob_list = []

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in test_loader:

    # Creating new variables for the hidden state
    h = tuple([each.data for each in h])

    if train_on_gpu:
        inputs, labels = inputs.cuda(), labels.cuda()

    # get predicted outputs
    output, h = net(inputs, h)

    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())

    true_list.extend(list(labels.cpu().numpy()))
    pred_list.extend(list(pred.cpu().detach().numpy().astype(int)))
    prob_list.extend(list(output.cpu().detach().numpy()))

    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = (
        np.squeeze(correct_tensor.numpy())
        if not train_on_gpu
        else np.squeeze(correct_tensor.cpu().numpy())
    )
    num_correct += np.sum(correct)



# Classification report
print(classification_report(true_list, pred_list, digits=3))

fpr, tpr, thresholds = metrics.roc_curve(true_list, prob_list, pos_label=1)
auroc = metrics.auc(fpr, tpr)
print("Area under ROC curve:", auroc)

pr, re, thresholds = metrics.precision_recall_curve(true_list, prob_list, pos_label=1)
auprc = metrics.auc(re, pr)
print("Area under PR curve:", auprc)

