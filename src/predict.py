import torch

from config import *
from model import TranslateModel
from tokenizer import EnglishTokenizer, ChineseTokenizer

def predict_batch(model, inputs, tokenizer, device):
    model.eval()
    with torch.no_grad():
        batch_size = inputs.shape[0]
        context_vectors = model.encoder(inputs)

        decoder_hidden_output = context_vectors.unsqueeze(0)
        decoder_input = torch.full(size=(batch_size, 1), fill_value=tokenizer.start_token_id, device=device)

        generated_ids = []
        for i in range(SEQ_LEN):
            decoder_output, decoder_hidden_output = model.decoder(decoder_input, decoder_hidden_output)
            next_token_id = torch.argmax(decoder_output, dim=-1)
            generated_ids.append(next_token_id)
            decoder_input = next_token_id

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