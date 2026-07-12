import pandas as pd
import jieba
from sklearn.model_selection import train_test_split

from config import *
from tokenizer import JiebaTokenizer

def preprocess():
    print('The preprocessing of data is starting...')

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA_DIR/RAW_DATA_FILE, usecols=['label', 'review'], encoding='utf-8')
    df = df.dropna(subset=['label', 'review']).copy()
    df['review'] = df['review'].astype(str)

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])
    train_df = train_df.copy()
    test_df = test_df.copy()

    JiebaTokenizer.build_vocab(train_df['review'].tolist(), MODEL_DIR/VOCAB_FILE)

    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)

    train_df['review'] = train_df['review'].apply(lambda x: tokenizer.encode(x, SEQ_LEN))
    test_df['review'] = test_df['review'].apply(lambda x: tokenizer.encode(x, SEQ_LEN))

    train_df.to_json(PROCESSED_DATA_DIR/TRAIN_DATA_FILE, orient='records', lines=True)
    test_df.to_json(PROCESSED_DATA_DIR/TEST_DATA_FILE, orient='records', lines=True)

    print('The preprocessing of data is done.')

if __name__ == '__main__':
    preprocess()
