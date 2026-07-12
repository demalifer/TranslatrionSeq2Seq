import torch

from config import *
from model import ReviewAnalysisModel
from tokenizer import JiebaTokenizer

def predict_batch(model, inputs):
    model.eval()
    with torch.no_grad():
        output = model(inputs)
    batch_results = torch.sigmoid(output)
    return batch_results.tolist()

def predict(text, model, tokenizer, device):

    ids = tokenizer.encode(text, SEQ_LEN)

    input = torch.tensor([ids], dtype=torch.long).to(device)

    result = predict_batch(model, input)

    return result[0]

def run_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)
    print('vocabulary load success!')

    model = ReviewAnalysisModel(vocab_size=tokenizer.vocab_size, padding_idx=tokenizer.pad_token_id).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / BEST_MODEL))
    print('model load success!')

    print('Welcome to INTELEGER sentiment analysis model! print q or quit to exit...')
    input_history = ''
    while True:
        user_input = input('> ')
        if user_input.strip() in ['q', 'quit']:
            print('bye!')
            break
        elif user_input.strip() == '':
            print('please input valid content!')
            continue

        result = predict(user_input, model, tokenizer, device)
        if result >= 0.5:
            print(f'positive review: (confidence level: {result})')
        else:
            print(f'negative review: (confidence level: {1 - result})')

if __name__ == '__main__':
    run_predict()