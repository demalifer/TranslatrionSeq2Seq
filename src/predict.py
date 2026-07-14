import torch

from config import *
from model import TranslateModel
from tokenizer import EnglishTokenizer, ChineseTokenizer

def predict_batch(model, inputs, tokenizer, device):
    model.eval()
    with torch.no_grad():
        batch_size = inputs.shape[0]
        encoder_outputs, context_vectors = model.encoder(inputs)

        decoder_hidden_output = context_vectors.unsqueeze(0)
        decoder_input = torch.full(size=(batch_size, 1), fill_value=tokenizer.start_token_id, device=device)

        generated_ids = []
        is_finished = torch.full(size=[batch_size], fill_value=False, device=device)
        for i in range(SEQ_LEN):
            decoder_output, decoder_hidden_output = model.decoder(decoder_input, decoder_hidden_output, encoder_outputs)
            next_token_id = torch.argmax(decoder_output, dim=-1)
            generated_ids.append(next_token_id)
            decoder_input = next_token_id
            is_finished |= (next_token_id.squeeze(1) == tokenizer.end_token_id)
            if is_finished.all():
                break

    generated_list = torch.cat(generated_ids, dim=1).tolist()
    for i, sentence_ids in enumerate(generated_list):
        if tokenizer.end_token_id in sentence_ids:
            eos_pos = sentence_ids.index(tokenizer.end_token_id)
            generated_list[i] = sentence_ids[:eos_pos]

    return generated_list

def predict(text, model, cn_tokenizer, en_tokenizer, device):

    ids = cn_tokenizer.encode(text)

    input = torch.tensor([ids], dtype=torch.long).to(device)

    result = predict_batch(model, input, en_tokenizer, device)

    return en_tokenizer.decode(result[0])

def run_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    en_tokenizer = EnglishTokenizer.from_vocab(MODEL_DIR/EN_VOCAB_FILE)
    cn_tokenizer = ChineseTokenizer.from_vocab(MODEL_DIR/CN_VOCAB_FILE)
    print('vocabulary load success!')

    model = TranslateModel(cn_tokenizer.vocab_size, en_tokenizer.vocab_size, cn_tokenizer.pad_token_id, en_tokenizer.pad_token_id).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / BEST_MODEL))
    print('model load success!')

    print('Welcome to INTELEGER CH->EN translate model! print q or quit to exit...')
    while True:
        user_input = input('> please input a Chinese sentence: ')
        if user_input.strip() in ['q', 'quit']:
            print('bye!')
            break
        elif user_input.strip() == '':
            print('please input valid content!')
            continue

        result = predict(user_input, model, cn_tokenizer, en_tokenizer, device)
        print('In English: ', result)

if __name__ == '__main__':
    run_predict()