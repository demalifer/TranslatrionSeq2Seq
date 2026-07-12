import jieba
from config import *
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer

class BaseTokenizer():
    unk_token = UNK_TOKEN
    pad_token = PAD_TOKEN
    start_token = START_TOKEN
    end_token = END_TOKEN

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2id = {word : id for id, word in enumerate(vocab_list)}
        self.id2word = {id : word for id, word in enumerate(vocab_list)}
        self.unk_token_id = self.word2id[self.unk_token]
        self.pad_token_id = self.word2id[self.pad_token]
        self.start_token_id = self.word2id[self.start_token]
        self.end_token_id = self.word2id[self.end_token]

    @classmethod
    def tokenize(cls, text) -> list[str]:
        pass

    def encode(self, text, mark=False):
        tokens = self.tokenize(text)
        
        if mark:
            tokens = [self.start_token] + tokens + [self.end_token]

        ids = [self.word2id.get(token, self.unk_token_id) for token in tokens]
        return ids

    @classmethod
    def build_vocab(cls, sentences, vocab_file_path):
        vocab_set = set()
        for sentence in sentences:
            vocab_set.update(cls.tokenize(sentence))
        vocab_list = [cls.pad_token, cls.unk_token, cls.start_token, cls.end_token] + list(vocab_set)

        print("The size of vocabulary:", len(vocab_list))

        with open(vocab_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_file_path):
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            vocab_list = [token.strip() for token in f.readlines()]
        tokenizer = cls(vocab_list)
        return tokenizer
    
class ChineseTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, text) -> list[str]:
        return list(text)
    
class EnglishTokenizer(BaseTokenizer):
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    @classmethod
    def tokenize(cls, text) -> list[str]:
        return cls.tokenizer.tokenize(text)

    def decode(self, ids):
        tokens = [self.id2word[id] for id in ids]
        return self.detokenizer.detokenize(tokens)

if __name__ == '__main__':
    en_tokenizer = EnglishTokenizer.from_vocab(MODEL_DIR / EN_VOCAB_FILE)
    cn_tokenizer = ChineseTokenizer.from_vocab(MODEL_DIR / CN_VOCAB_FILE)
    print('the size of English vocabulary:', len(en_tokenizer.vocab_list))
    print('the size of Chinese vocabulary:', len(cn_tokenizer.vocab_list))
    print('UNK_TOKEN: ', cn_tokenizer.unk_token)
    print('PAD_TOKEN: ', en_tokenizer.pad_token)
    print('START_TOKEN: ', en_tokenizer.start_token)
    print('END_TOKEN: ', en_tokenizer.end_token)
    print(cn_tokenizer.encode('自然语言处理'))
    print(en_tokenizer.encode('natural language process', mark=True))