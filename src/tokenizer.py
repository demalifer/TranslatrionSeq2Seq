import jieba
from config import *

class JiebaTokenizer():
    unk_token = UNK_TOKEN
    pad_token = PAD_TOKEN

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2id = {word : id for id, word in enumerate(vocab_list)}
        self.id2word = {id : word for id, word in enumerate(vocab_list)}
        self.unk_token_id = self.word2id[self.unk_token]
        self.pad_token_id = self.word2id[self.pad_token]

    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    def encode(self, text, seq_len):
        tokens = self.tokenize(text)

        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens) < seq_len:
            tokens = tokens + [self.pad_token] * (seq_len - len(tokens))

        ids = [self.word2id.get(token, self.unk_token_id) for token in tokens]
        return ids

    @classmethod
    def build_vocab(cls, sentences, vocab_file_path):
        vocab_set = set()
        for sentence in sentences:
            vocab_set.update(jieba.lcut(sentence))
        vocab_list = [cls.pad_token, cls.unk_token] + list(vocab_set)

        print("The size of vocabulary:", len(vocab_list))

        with open(vocab_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_file_path):
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            vocab_list = [token.strip() for token in f.readlines()]
        tokenizer = cls(vocab_list)
        return tokenizer

if __name__ == '__main__':
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)
    print('the size of vocabulary:', len(tokenizer.vocab_list))
    print('UNK_TOKEN: ', tokenizer.unk_token)
    print('PAD_TOKEN: ', tokenizer.pad_token)
    print(tokenizer.encode('自然语言处理', seq_len=SEQ_LEN))