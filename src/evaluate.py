import torch
from config import *
from model import ReviewAnalysisModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import JiebaTokenizer
from tqdm import tqdm

def evaluate(model, dataloader, device):
    correct_count = 0
    total_count = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Evaluating: '):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_results = predict_batch(model, inputs)
            for target, result in zip(targets, batch_results):
                total_count += 1
                result = 1 if result > 0.5 else 0
                if result == target:
                    correct_count += 1
    correct_rate = correct_count / total_count
    return correct_rate

def run_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)
    print('vocabulary load success!')

    model = ReviewAnalysisModel(vocab_size=tokenizer.vocab_size, padding_idx=tokenizer.pad_token_id).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / BEST_MODEL))
    print('model load success!')

    test_dataloader = get_dataloader(train=False)
    correct_rate = evaluate(model, test_dataloader, device)

    print('evaluate result:')
    print('correct rate: ', correct_rate)

if __name__ == '__main__':
    run_evaluate()
