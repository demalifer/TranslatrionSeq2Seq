import torch
from torch import nn, optim
from tqdm import tqdm

from config import *
from dataset import get_dataloader
from model import TranslateModel

from torch.utils.tensorboard import SummaryWriter
import time
from tokenizer import ChineseTokenizer, EnglishTokenizer

def train_one_epoch(model, train_loader, loss , optimizer, device):
    model.train()

    total_loss = 0
    for inputs, targets in tqdm(train_loader, desc='Training: '):
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        context_vectors = model.encoder(inputs)

        decoder_inputs = targets[:, :-1]
        decoder_targets = targets[:, 1:]
        decoder_h0 = context_vectors.unsqueeze(0)
        decoder_outputs, _ = model.decoder(decoder_inputs, decoder_h0)

        loss_value = loss(decoder_outputs.transpose(1, 2), decoder_targets)
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss_value.item()

    return total_loss / len(train_loader)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_dataloader(train=True)

    cn_tokenizer = ChineseTokenizer.from_vocab(MODEL_DIR/CN_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(MODEL_DIR/EN_VOCAB_FILE)

    model = TranslateModel(cn_tokenizer.vocab_size, en_tokenizer.vocab_size, cn_tokenizer.pad_token_id, en_tokenizer.pad_token_id).to(device)
    loss = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(log_dir=LOG_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    min_loss = float('inf')
    for epoch in range(EPOCHS):
        print('='*15, f'EPOCH {epoch+1}', '='*15)
        this_loss = train_one_epoch(model, train_loader, loss, optimizer, device)
        print('the loss of this epoch is : ', this_loss)

        writer.add_scalar('loss', this_loss, epoch + 1)

        if this_loss < min_loss:
            min_loss = this_loss
            torch.save(model.state_dict(), MODEL_DIR / BEST_MODEL)
            print('The best model has been saved!')
        else:
            print('This model is not the best, and it has not been saved!')
    writer.close()

if __name__ == '__main__':
    train()
