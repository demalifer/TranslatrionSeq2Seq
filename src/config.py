from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

RAW_DATA_FILE = 'cmn.txt'
TRAIN_DATA_FILE = 'train.jsonl'
TEST_DATA_FILE = 'test.jsonl'
EN_VOCAB_FILE = 'en_vocab.txt'
CN_VOCAB_FILE = 'cn_vocab.txt'
BEST_MODEL = 'best_model.pt'

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
START_TOKEN = '<sos>'
END_TOKEN = '<eos>'

SEQ_LEN = 128
BATCH_SIZE = 64
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 256

LEARNING_RATE = 0.001
EPOCHS = 100