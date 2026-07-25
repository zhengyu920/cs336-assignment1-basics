import numpy as np
from tokenizer import Tokenizer
from typing import BinaryIO
import pickle

def encode_data(file: BinaryIO, tokenizer: Tokenizer) -> np.ndarray:
    tokens = []
    for t in tokenizer.encode_iterable(file):
        tokens.append(t)
    return np.array(tokens, dtype=np.uint16)


with open('bpe_ts_vocab.pkl', 'rb') as f:
    vocab_ts = pickle.load(f)
with open('bpe_ts_merges.pkl', 'rb') as f:
    merges_ts = pickle.load(f)

with open('bpe_owt_vocab.pkl', 'rb') as f:
    vocab_owt = pickle.load(f)
with open('bpe_owt_merges.pkl', 'rb') as f:
    merges_owt = pickle.load(f)

ST = '<|endoftext|>'

ts_tokenizer = Tokenizer(vocab_ts, merges_ts, [ST])
owt_tokenizer = Tokenizer(vocab_owt, merges_owt, [ST])

TS_FILE_PATH = 'data/TinyStoriesV2-GPT4-train.txt'
TS_VAL_FILE_PATH = 'data/TinyStoriesV2-GPT4-valid.txt'
TS_ENCODE_FILE_PATH = 'data/ts_train_encoded.npy'
TS_VAL_ENCODE_FILE_PATH = 'data/ts_val_encoded.npy'

OWT_FILE_PATH = 'data/owt_train.txt'
OWT_VAL_FILE_PATH = 'data/owt_valid.txt'
OWT_ENCODE_FILE_PATH = 'data/owt_train_encoded.npy'
OWT_VAL_ENCODE_FILE_PATH = 'data/owt_val_encoded.npy'

def encode_file(data_file_path, np_file_path, tokenizer):
    with open(data_file_path, 'r') as f:
        val = encode_data(f, tokenizer)
        np.save(np_file_path, val)

encode_file(TS_FILE_PATH, TS_ENCODE_FILE_PATH, ts_tokenizer)
encode_file(TS_VAL_FILE_PATH, TS_VAL_ENCODE_FILE_PATH, ts_tokenizer)
encode_file(OWT_FILE_PATH, OWT_ENCODE_FILE_PATH, owt_tokenizer)
encode_file(OWT_VAL_FILE_PATH, OWT_VAL_ENCODE_FILE_PATH, owt_tokenizer)