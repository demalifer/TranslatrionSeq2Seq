import torch
import torch.nn as nn

from config import *

class TranslateEncoder(nn.Module):
    def __init__(self, vocab_size, padding_idx=0):
        super(TranslateEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)

    def forward(self, input):
        embed = self.embedding(input)
        output, _ = self.gru(embed)
        indices = torch.arange(output.shape[0])
        lengths = (input != self.embedding.padding_idx).sum(dim=1)
        features = output[indices, lengths - 1]
        return features

class TranslateDecoder(nn.Module):
    def __init__(self, vocab_size, padding_idx=0):
        super(TranslateDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=vocab_size)

    def forward(self, input, h0=None):
        embed = self.embedding(input)
        output, hn = self.gru(embed, h0)
        output = self.linear(output)
        return output, hn

class TranslateModel(nn.Module):
    def __init__(self, cn_vocab_size, en_vocab_size, cn_padding_idx=0, en_padding_idx=0):
        super(TranslateModel, self).__init__()
        self.encoder = TranslateEncoder(cn_vocab_size, cn_padding_idx)
        self.decoder = TranslateDecoder(en_vocab_size, en_padding_idx)

if __name__ == '__main__':
    vocab_size = 1000
    model = TranslateModel(1000, 1024)
    print(model.encoder)
    print(model.decoder)
