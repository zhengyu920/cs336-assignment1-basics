from train_bpe import train_bpe
import pickle

MAX_VOCAB_SIZE = 10000
S_T = ['<|endoftext|>']

FILE_PATH = 'data/TinyStoriesV2-GPT4-train.txt'

vocab, merges = train_bpe(FILE_PATH, MAX_VOCAB_SIZE, S_T, num_processes=8)

with open('bpe_ts_vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
with open('bpe_ts_merges.pkl', 'wb') as f:
    pickle.dump(merges, f)