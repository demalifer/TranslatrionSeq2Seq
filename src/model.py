import torch
import torch.nn as nn

from config import *

class Attention(nn.Module):
    def forward(self, decoder_hidden_outputs, encoder_outputs):
        attention_scores = torch.bmm(decoder_hidden_outputs, encoder_outputs.transpose(1, 2))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vector = torch.bmm(attention_weights, encoder_outputs)
        return context_vector

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
        return output, features

class TranslateDecoder(nn.Module):
    def __init__(self, vocab_size, padding_idx=0):
        super(TranslateDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.attention = Attention()
        self.linear = nn.Linear(in_features=HIDDEN_SIZE * 2, out_features=vocab_size)

    def forward(self, input, h0, encoder_outputs):
        embed = self.embedding(input)
        output, hn = self.gru(embed, h0)
        context_vector = self.attention(output, encoder_outputs)
        combined = torch.cat((output, context_vector), dim=-1)
        output = self.linear(combined)
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
