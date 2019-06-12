import torch
import torch.nn as nn
import torch.nn.functional as F

#This file contains all the self-defined layers that we constructed 
#reference: CS224N Final Project Code
class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, n_src_vocab, hidden_size, drop_prob, n_src_char, embeds=None):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        # self.embed = nn.Embedding.from_pretrained(word_vectors)      # changed

        #emb size
        emb_size = hidden_size
        #For this project we keep the final embedding size the same as the original embedding size

        ### start our code:
        self.n_src_vocab = n_src_vocab
        self.n_src_char = n_src_char 
        self.hidden_size = hidden_size
        
        self.embed_size = emb_size
        self.word_embed = nn.Embedding(n_src_vocab, emb_size, padding_idx=0)
        if embeds is not None:
            self.word_embed.weight.data.copy_(torch.from_numpy(embeds))
            self.word_embed.weight.requires_grad = False

        #Thsi is the character embedding, with the same embedding size as the word embedding
        self.char_embed = nn.Embedding(len(n_src_char), emb_size , padding_idx = n_src_char['<pad>'])

        #need to be the same as what we defined in pad_to_long#################
        self.max_word_len = 40   # args.py (char_limit)
        self.kernel_size = 5

        self.cnn = CNN(emb_size, self.max_word_len, emb_size, self.kernel_size)
        
        #Apply the dropout here
        self.dropout = nn.Dropout(self.drop_prob)

        self.proj = nn.Linear(2 * emb_size, self.hidden_size, bias=False)

        ### end our code:

        # self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        # torch.nn.Linear(in_features, out_features, bias=True)

        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, word_x, char_x):

        # emb = self.embed(x)   # (batch_size, seq_len, embed_size)

        ### start our code:
        # torch.cat(tensors, dim=0, out=None)

        #print("word_vectors: ", self.word_vectors.size())   # torch.Size([88714, 300])
        #print("char_vectors: ", self.char_vectors.size())   # torch.Size([1376, 64])

        w_emb = self.word_embed(word_x)
        #print("w_emb size:", w_emb.size())   # torch.Size([64, 338, 300])

        # ch_emb = self.char_embed(char_x)
        # print("ch_emb size:", ch_emb.size())

        ###
        x_split = torch.split(char_x, 1, dim = 0)

        ch_emb = []
        for x in x_split:
            x = torch.squeeze(x, dim = 0)
            x_emb = self.char_embed(x)
            x_reshape = x_emb.permute(0, 2, 1)
            x_conv_out = self.cnn(x_reshape)
            ch_emb.append(x_conv_out)

        ch_emb = torch.stack(ch_emb, dim=0)
        #print("ch_emb size:", ch_emb.size())   # [64, 307, 300]


        ###
        # x = torch.flatten(input = char_x, end_dim = 1)
        # x = torch.squeeze(x, dim = 0)
        # x_emb = self.char_embed(x)
        # x_reshape = x_emb.permute(0, 2, 1)
        # x_conv_out = self.cnn(x_reshape)
        # print("x_conv_out size:", x_conv_out.size())


        # ch_emb = self.cnn(ch_emb)
        # print("ch_emb size:", ch_emb.size())

        ### We changed dim = 2 because we are supposed to
        ### concatenated the embedding
        ### not the sequence length

        emb = torch.cat((w_emb, ch_emb), dim = 2)
        ###print("concat emb size:", emb.size())   # [64, 614, 300]

        ### end our code:

        emb = self.dropout(emb)
                # training – apply dropout if is True [?]
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

### start our code
class CNN(nn.Module):
    """Implement CNN class

    Args:
    """
    def __init__(self, char_embed_size, max_word_len, word_embed_size, kernel_size = 5):
        """ Initialize CNN network

        @param char_embed_size: size for the character embedding
        @param max_word_len: maximum word length
        @param word_embed_size: size for the word embedding
        @param kernel_size: kernel size, default = 5

        """
        super(CNN, self).__init__()

        # e_char, e_word, m_word, k (kernel_size)
        self.e_char = char_embed_size
        self.k = kernel_size
        self.m_word = max_word_len
        self.e_word = word_embed_size   # f

        # default
        self.ConvLayer = None
        self.MaxPool = None

        # 1st dim:
        self.ConvLayer = nn.Conv1d(self.e_char, self.e_word, self.k, bias = True)
                                   # (char_embed_size, word_embed_size, kernel_size)
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

        # torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.MaxPool = nn.MaxPool1d(self.m_word - self.k + 1)

    def forward(self, x_reshaped : torch.Tensor) -> torch.Tensor:
        """
        Map from x_reshaped to x_conv out

        @param: torch tensor x_reshaped

        @returns x_conv_out
        """

        x_conv = self.ConvLayer(x_reshaped)
        x_conv_relu = F.relu(x_conv)
        x_conv_out = self.MaxPool(x_conv_relu)
        x_conv_out = torch.squeeze(x_conv_out, dim = 2)

        return x_conv_out

### end our code



class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x