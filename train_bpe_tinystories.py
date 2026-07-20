from train_bpe import train_bpe
import pickle

MAX_VOCAB_SIZE = 10000
S_T = ['<|endoftext|>']

FILE_PATH = 'data/TinyStoriesV2-GPT4-valid.txt'

vocab, merges = train_bpe(FILE_PATH, MAX_VOCAB_SIZE, S_T)

pickle.dump(vocab, 'bpe_ts_vocab.pkl')
pickle.dump(merges, 'bpe_ts_merges.pkl')