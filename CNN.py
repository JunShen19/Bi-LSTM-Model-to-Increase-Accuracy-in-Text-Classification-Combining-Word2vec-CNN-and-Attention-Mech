from multiprocessing.spawn import prepare
import numpy as np
import pandas as pd
import tool 

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

# dataset
from torch.utils.data import TensorDataset, DataLoader
imdb_data = tool.data_preprocessing(r'D:\JunShen\nlpPractice\IMDB Dataset.csv')
# 
from sklearn.model_selection import train_test_split
X = imdb_data['review']
y = imdb_data['sentiment']
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2)
print(f'shape of train data is {x_train.shape}')
print(f'shape of test data is {x_test.shape}')

# create a common word dataset
word_list = []
for sent in x_train:
    for word in sent:
        word_list.append(word)

from collections import Counter
corpus = Counter(word_list)
# sorting on the basis of most common words
corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
# creating a dict
onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

final_list_train,final_list_test = [],[]
for sent in x_train:
    final_list_train.append([onehot_dict[word] for word in sent
        if word in onehot_dict.keys()])
for sent in x_test:
    final_list_test.append([onehot_dict[word] for word in sent
        if word in onehot_dict.keys()])

x_train = final_list_train
x_test = final_list_test

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

x_train_pad = padding_(x_train,500)
x_test_pad = padding_(x_test,500)

# create datasets
train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train.values))
test_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test.values))

# Preparing your data for training with DataLoaders
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print('Sample input: \n', sample_y)

## settings and train model for Word2Vec
from gensim.models import Word2Vec
def word2vec_train_return_df(imdb_data):
    seed = 42
    sg = 1 # CBOW or Skip-gram
    window_size = 3
    vector_size = 50 # embedding dim
    min_count = 1
    workers = 5
    epochs = 5
    batch_words = 10000

    model = Word2Vec(
        imdb_data['review'],
        min_count=min_count,
        vector_size=vector_size,
        workers=workers,
        epochs=epochs,
        window=window_size,
        sg=sg,
        seed=seed,
        batch_words=batch_words
    )
    # create embedding_df
    emb_df = (
        pd.DataFrame(
            [model.wv.get_vector(str(n)) for n in model.wv.key_to_index],
            index = model.wv.key_to_index
        )
    )

    return emb_df

emb_df = word2vec_train_return_df(imdb_data)

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
num_epochs = 4
learning_rate = 0.001

vocab_size = min(len(emb_df)+1, len(onehot_dict))
output_size = 1 # positive or negative

embedding_dim = 50

# prepare pretrained weight matrix
emb_matrix = np.zeros((vocab_size+1, embedding_dim))
for v, i in onehot_dict.items():
    emb_matrix[i] = emb_df.loc[emb_df.index == v, :].values
del emb_df
emb_matrix = torch.FloatTensor(emb_matrix)

config = {}
# set oarameters:
config['vocab_size'] = 1000
config['max_text_len'] = 500
config['batch_size'] = 64
config['embedding_dims'] = 50
config['dropout_rate'] = 0.2
config['window_sizes'] = [1,2,3]
config['feature_size'] = 100

# define model
class TextConvNet(nn.Module):
    def __init__(self, config):
        super(TextConvNet, self).__init__()
        self.dropout_rate = config['dropout_rate']
        self.num_class = 1
        self.window_sizes = config['window_sizes']

        self.embedding = nn.Embedding.from_pretrained(emb_matrix)
        self.embedding.weight.requires_grad = True

        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config['embedding_dims'],
            out_channels=config['feature_size'],
            kernel_size=h),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=config['max_text_len']-h+1))
            for h in config['window_sizes']
        ])
        self.fc = nn.Linear(in_features=config['feature_size']*len(self.window_sizes),
        out_features=self.num_class)

    def forward(self, x):
        embed_x = self.embedding(x) # -> n, 500, 50
        # reshape -> batch_suze, embedding_size, text_len -> 64, 50, 500
        embed_x = embed_x.permute(0,2,1)
        out = [conv(embed_x) for conv in self.convs]

        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))

        out = F.dropout(input=out, p=self.dropout_rate)
        out = self.fc(out)
        
        return out

model = TextConvNet(config).to(device)

# define loss function and optimzer
import torch.optim as optim

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

EPOCHS = 5
model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch.unsqueeze(1).float())
        acc = binary_acc(y_pred, y_batch.unsqueeze(1).float())
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    print(model.embedding.weight)
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

y_pred_list = []
y_test_list = []
model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())
        y_test_list.append(y_batch)

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
y_pred_list = [item for sublist in y_pred_list for item in sublist]

y_test_list = [a.squeeze().tolist() for a in y_test_list]
y_test_list = [item for sublist in y_test_list for item in sublist]

from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_test_list, y_pred_list)
print(classification_report(y_test_list, y_pred_list))
