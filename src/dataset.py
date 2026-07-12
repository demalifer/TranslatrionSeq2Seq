import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from config import *

class TranslateDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input = torch.tensor(self.data[index]['cn'], dtype=torch.long)
        target = torch.tensor(self.data[index]['en'], dtype=torch.long)
        return input, target

def collate_fn(batch):
    input_tensors = [item[0] for item in batch]
    target_tensors = [item[1] for item in batch]
    input_batch_tensor = pad_sequence(input_tensors, batch_first=True)
    target_batch_tensor = pad_sequence(target_tensors, batch_first=True)
    return input_batch_tensor, target_batch_tensor

def get_dataloader(train=True):
    path = PROCESSED_DATA_DIR / (TRAIN_DATA_FILE if train else TEST_DATA_FILE)
    dataset = TranslateDataset(path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    return dataloader

if __name__ == '__main__':
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)

    for input, target in train_dataloader:
        print(input.shape, target.shape)
        break
