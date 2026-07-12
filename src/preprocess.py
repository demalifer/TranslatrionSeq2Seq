import pandas as pd
import jieba
from sklearn.model_selection import train_test_split

from config import *
from tokenizer import ChineseTokenizer, EnglishTokenizer

def preprocess():
    print('The preprocessing of data is starting...')

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA_DIR/RAW_DATA_FILE, sep='\t', usecols=[0, 1], names=['en', 'cn'], encoding='utf-8').dropna()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    ChineseTokenizer.build_vocab(train_df['cn'].tolist(), MODEL_DIR/CN_VOCAB_FILE)
    EnglishTokenizer.build_vocab(train_df['en'].tolist(), MODEL_DIR/EN_VOCAB_FILE)
    
    cn_tokenizer = ChineseTokenizer.from_vocab(MODEL_DIR/CN_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(MODEL_DIR/EN_VOCAB_FILE)

    train_df['cn'] = train_df['cn'].apply(lambda x: cn_tokenizer.encode(x, mark=False))
    train_df["en"] = train_df["en"].apply(lambda x: en_tokenizer.encode(x, mark=True))

    test_df['cn'] = test_df['cn'].apply(lambda x: cn_tokenizer.encode(x, mark=False))
    test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x, mark=True))

    train_df.to_json(PROCESSED_DATA_DIR/TRAIN_DATA_FILE, orient='records', lines=True)
    test_df.to_json(PROCESSED_DATA_DIR/TEST_DATA_FILE, orient='records', lines=True)

    print('The preprocessing of data is done.')

if __name__ == '__main__':
    preprocess()
