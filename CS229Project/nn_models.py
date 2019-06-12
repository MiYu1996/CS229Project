import torch
import torch.nn as nn
from torch.nn import functional as F

## Below models: courtesy of the original github repo

class GMP(nn.Module):
    def __init__(self, n_src_vocab, embeds=None, dropout=0.3):
        super(GMP, self).__init__()
        label_size = 1
        emb_size = 300
        hidden = 128

        self.word_embeddings = nn.Embedding(n_src_vocab, emb_size, padding_idx=0)
        if embeds is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embeds))
            self.word_embeddings.weight.requires_grad = False

        self.rnn = nn.GRU(emb_size, hidden, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(2*hidden, label_size)
        self.drop = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        embeds = self.word_embeddings(src)
        embeds = self.drop(embeds)
        embeds, hid = self.rnn(embeds)
        embeds = embeds.max(1)[0]
        x = embeds
        x = self.linear(x)
        return x

    def get_trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)


    
    
class GRUCnn(nn.Module):
    def __init__(self, n_src_vocab, embeds=None, dropout=0.3):
        super(GRUCnn, self).__init__()
        label_size = 1
        emb_size = 300
        hid1 = 128
        hid2 = 128
        
        self.word_embeddings = nn.Embedding(n_src_vocab, emb_size, padding_idx=0)
        if embeds is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embeds))
            self.word_embeddings.weight.requires_grad = False

        self.conv = nn.Conv1d(2*hid1, hid2, 2)
        self.rnn = nn.GRU(emb_size, hid1, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(hid2, label_size)
        self.drop = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        embeds = self.word_embeddings(src)
        embeds = self.drop(embeds)
        embeds, hid = self.rnn(embeds)
        embeds = embeds.permute(0, 2, 1)

        cnn = self.conv(embeds)
        cnn = self.activation(cnn)
        cnn = cnn.permute(0, 2, 1)
        cnn = cnn.max(1)[0]
        x = self.linear(cnn)
        return x

    def get_trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)

class SelfAttention(nn.Module):
    """ Self-matching attention layer (inspiration from RNet paper)
    Directly match question-aware passage representation against itself
    """

    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.WP = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_out = nn.Linear(in_features = 3 * self.hidden_size, out_features = self.hidden_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, g):
        # Multiplicative self-attention
        Wh = self.WP(g)
        s_t = torch.bmm(Wh, g.permute([0, 2, 1]))
        a_t = F.softmax(s_t, 1).squeeze()
        c_t = torch.bmm(a_t, g).squeeze()
        out = torch.cat((g, c_t, g * c_t), dim = 2)
        out = self.linear_out(out)
        out = self.relu(out)
        return out

class GRUCnnAtt(nn.Module):
    def __init__(self, n_src_vocab, embeds=None, dropout=0.3):
        super(GRUCnnAtt, self).__init__()
        label_size = 1
        emb_size = 300
        hid1 = 128
        hid2 = 128
        
        self.word_embeddings = nn.Embedding(n_src_vocab, emb_size, padding_idx=0)
        if embeds is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embeds))
            self.word_embeddings.weight.requires_grad = False

        self.conv = nn.Conv1d(2*hid1, hid2, 2)
        self.rnn = nn.GRU(emb_size, hid1, num_layers=1, bidirectional=True)
        self.self_attention = SelfAttention(hid2)
        self.linear = nn.Linear(hid2, label_size)
        self.drop = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        embeds = self.word_embeddings(src)
        embeds = self.drop(embeds)
        embeds, hid = self.rnn(embeds)
        embeds = embeds.permute(0, 2, 1)

        cnn = self.conv(embeds)
        cnn = self.activation(cnn)
        cnn = cnn.permute(0, 2, 1)
        cnn = self.self_attention(cnn)
        cnn = cnn.max(1)[0]
        x = self.linear(cnn)
        return x

    def get_trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)

# Architecture proposed by paper Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling
# Hyperparameters closely follows the paper's suggestions
class bLSTM2DCNN(nn.Module):
    def __init__(self, n_src_vocab, embeds=None, emb_dprob = 0.3, rnn_dprob = 0.2, cnn_dprob = 0.3):
        super(bLSTM2DCNN, self).__init__()
        self.label_size = 1
        self.emb_size = 300
        self.rnn_hidden_size = 100
        self.conv_filters = 60
        self.emb_dprob = emb_dprob
        self.rnn_dprob = rnn_dprob
        self.cnn_dprob = cnn_dprob
        self.n_src_vocab = n_src_vocab

        self.word_embeddings = nn.Embedding(self.n_src_vocab, self.emb_size, padding_idx=0)
        if embeds is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embeds))
            self.word_embeddings.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout(p = self.emb_dprob)
        self.rnn = nn.GRU(self.emb_size, self.rnn_hidden_size, num_layers=1, bidirectional=True)
        self.rnn_dropout = nn.Dropout(p = self.rnn_dprob)
        self.conv = nn.Conv2d(1, self.conv_filters, 3, stride = 1)
        self.maxpool = nn.MaxPool2d(2)
        self.cnn_dropout = nn.Dropout(p = self.cnn_dprob)
        self.linear = nn.Linear(self.rnn_hidden_size - 1, self.label_size)

    def forward(self, src):
        embeds = self.word_embeddings(src)
        embeds = self.embedding_dropout(embeds)
        x, _ = self.rnn(embeds)
        x = self.rnn_dropout(x)
        # print("after rnn shape", x.shape)
        x = x.permute(0, 2, 1)
        # print("after rnn permute shape", x.shape)
        x = x.unsqueeze(1)
        # print("after unsqueeze shape", x.shape)
        cnn_out = self.conv(x)
        # print("immediately after conv", cnn_out.shape)
        cnn_out = self.maxpool(cnn_out)
        # print("maxpooled", cnn_out.shape)
        cnn_out = torch.mean(cnn_out, dim = 1)
        # print("after maxpool before squeeze", cnn_out.shape)
        cnn_out = cnn_out.squeeze()
        # print("after maxpool after squeeze", cnn_out.shape)
        cnn_out = cnn_out.permute(0, 2, 1)
        cnn_out = self.cnn_dropout(cnn_out)
        cnn_out = cnn_out.max(1)[0]
        x = self.linear(cnn_out)
        return x

    def get_trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)

# From kaggle kernel
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class doubleRNN(nn.Module):
    def __init__(self, n_src_vocab, embeds=None, dropout_prob = 0.3):
        super(doubleRNN, self).__init__()
        self.label_size = 1
        self.emb_size = 600
        self.rnn_hidden_size = 150
        self.dense_size = 4 * self.rnn_hidden_size
        self.dropout_prob = dropout_prob
        self.n_src_vocab = n_src_vocab

        self.word_embeddings = nn.Embedding(self.n_src_vocab, self.emb_size, padding_idx=0)
        if embeds is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embeds))
            self.word_embeddings.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        self.rnn = nn.LSTM(self.emb_size, self.rnn_hidden_size, num_layers=1, bidirectional=True)
        self.rnn_layer2 = nn.GRU(self.rnn_hidden_size * 2, self.rnn_hidden_size, num_layers=1, bidirectional=True)
        self.linear1 = nn.Linear(self.dense_size, self.dense_size)
        self.linear2 = nn.Linear(self.dense_size, self.dense_size)
        self.hidden_dropout = nn.Dropout(p = 0.2)
        self.linear_out = nn.Linear(self.dense_size, 1)
    
    def forward(self, src):
        embeds = self.word_embeddings(src)
        embeds = self.embedding_dropout(embeds)
        x, (_,_) = self.rnn(embeds)
        x, _ = self.rnn_layer2(x)
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        hidden= self.hidden_dropout(hidden)
        out = self.linear_out(hidden)
        return out 

    def get_trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)
    
class doubleRNNPlus(nn.Module):
    def __init__(self, n_src_vocab, embeds=None, dropout_prob = 0.3):
        super(doubleRNNPlus, self).__init__()
        self.label_size = 1
        self.emb_size = 300
        self.lstm_hidden_size = 120
        self.gru_hidden_size = 60
        # self.dense_size = 4 * self.rnn_hidden_size
        self.dropout_prob = dropout_prob
        self.n_src_vocab = n_src_vocab

        self.word_embeddings = nn.Embedding(self.n_src_vocab, self.emb_size, padding_idx=0)
        if embeds is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embeds))
            self.word_embeddings.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.2)
        self.rnn = nn.LSTM(self.emb_size, self.lstm_hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.rnn_layer2 = nn.GRU(self.lstm_hidden_size * 2, self.gru_hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.gru_hidden_size * 6, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(16, 1)
    
    def forward(self, src):
        embeds = self.word_embeddings(src)
        embeds = torch.unsqueeze(embeds.transpose(1, 2), 2)
        embeds = torch.squeeze(self.embedding_dropout(embeds)).transpose(1, 2)
        # embeds = self.embedding_dropout(embeds)
        x, (_,_) = self.rnn(embeds)
        x, x_gru = self.rnn_layer2(x)
        # print("x_gru", x_gru.shape)
        x_gru = x_gru.view(-1, self.gru_hidden_size * 2)
        # print("x_gru", x_gru.shape)
        avg_pool = torch.mean(x, 1)
        # print("avg_pool", avg_pool.shape)
        max_pool, _ = torch.max(x, 1)
        # print("max_pool", max_pool.shape)
        conc = torch.cat((x_gru, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out 

    def get_trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)

class  GRUCNNplus(nn.Module):
    def __init__(self, n_src_vocab, embeds=None, dropout=0.3):
        super(GRUCNNplus, self).__init__()
        label_size = 1
        emb_size = 300
        hid1 = 128
        hid2 = 128
        
        self.word_embeddings = nn.Embedding(n_src_vocab, emb_size, padding_idx=0)
        if embeds is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embeds))
            self.word_embeddings.weight.requires_grad = False

        self.conv = nn.Conv1d(2 * hid1, hid2, 2)
        self.rnn = nn.GRU(emb_size, hid1, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(hid2, label_size)
        self.drop = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        embeds = self.word_embeddings(src)
        embeds = self.drop(embeds)
        embeds, _ = self.rnn(embeds)
        embeds = embeds.permute(0, 2, 1)

        cnn = self.conv(embeds)
        cnn = self.activation(cnn)
        cnn = cnn.permute(0, 2, 1)
        cnn = cnn.max(1)[0]
        x = self.linear(cnn)
        return x

    def get_trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)

# Add self-attention after GRU
class doubleRNNAtt(nn.Module):
    def __init__(self, n_src_vocab, embeds=None, dropout_prob = 0.3):
        super(doubleRNNAtt, self).__init__()
        self.label_size = 1
        self.emb_size = 300
        self.lstm_hidden_size = 120
        self.gru_hidden_size = 60
        # self.dense_size = 4 * self.rnn_hidden_size
        self.dropout_prob = dropout_prob
        self.n_src_vocab = n_src_vocab

        self.word_embeddings = nn.Embedding(self.n_src_vocab, self.emb_size, padding_idx=0)
        if embeds is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embeds))
            self.word_embeddings.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.2)
        self.rnn = nn.LSTM(self.emb_size, self.lstm_hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.rnn_layer2 = nn.GRU(self.lstm_hidden_size * 2, self.gru_hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.self_attention = SelfAttention(self.gru_hidden_size * 2)
        self.dropout = nn.Dropout(0.2)
        # self.linear = nn.Linear(self.gru_hidden_size * 2, 16)
        # self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(self.gru_hidden_size * 2, 1)
    
    def forward(self, src):
        embeds = self.word_embeddings(src)
        embeds = torch.unsqueeze(embeds.transpose(1, 2), 2)
        embeds = torch.squeeze(self.embedding_dropout(embeds)).transpose(1, 2)
        # embeds = self.embedding_dropout(embeds)
        x, (_,_) = self.rnn(embeds)
        x, _ = self.rnn_layer2(x)
        x_att = self.self_attention(x)
        x_att = self.dropout(x_att)
        # x_att = self.relu(self.linear(x_att))
        out = self.out(x_att)
        out = out.max(1)[0]
        return out 

    def get_trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)