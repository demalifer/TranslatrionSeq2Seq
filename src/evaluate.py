import torch
from config import *
from model import TranslateModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import ChineseTokenizer, EnglishTokenizer
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

def evaluate(model, dataloader, tokenizer, device):
    references = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.tolist()
            batch_result = predict_batch(model, inputs, tokenizer, device)
            predictions.extend(batch_result)
            references.extend([[target[1:target.index(tokenizer.end_token_id)]] for target in targets])
    bleu_score = corpus_bleu(references, predictions)
    return bleu_score

def run_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cn_tokenizer = ChineseTokenizer.from_vocab(MODEL_DIR/CN_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(MODEL_DIR/EN_VOCAB_FILE)
    print('vocabulary load success!')

    model = TranslateModel(cn_tokenizer.vocab_size, en_tokenizer.vocab_size, cn_tokenizer.pad_token_id, en_tokenizer.pad_token_id).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / BEST_MODEL))
    print('model load success!')

    test_dataloader = get_dataloader(train=False)
    bleu = evaluate(model, test_dataloader, en_tokenizer, device)

    print('evaluate result:')
    print('BLEU Score: ', bleu)

if __name__ == '__main__':
    run_evaluate()
