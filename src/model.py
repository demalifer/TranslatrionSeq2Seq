import torch
import torch.nn as nn

from config import *

class ReviewAnalysisModel(nn.Module):
    def __init__(self, vocab_size, padding_idx=0):
        super(ReviewAnalysisModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=1)

    def forward(self, input):
        embed = self.embedding(input)
        output, _ = self.lstm(embed)
        indices = torch.arange(output.shape[0])
        lengths = (input != self.embedding.padding_idx).sum(dim=1)
        features = output[indices, lengths - 1]
        result = self.linear(features).squeeze(-1)
        return result

if __name__ == '__main__':
    vocab_size = 1000
    input = torch.randint(1000, size=(64, 5))
    model = ReviewAnalysisModel(vocab_size)
    output = model(input)
    print(output.size())
