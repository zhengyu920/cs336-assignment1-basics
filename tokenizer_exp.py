from sample_docs import sample_doc
from tokenizer import Tokenizer
import pickle
import statistics as stats

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
OWT_FILE_PATH = 'data/owt_train.txt'

def compression_ratio(tokenizer: Tokenizer, samples: list[str]) -> float:
    total_bytes = sum([len(doc.encode('utf-8')) for doc in samples])
    total_tokens = sum([len(tokenizer.encode(doc)) for doc in samples])
    return total_bytes/total_tokens

ts_result = []
owt_result = []

for i in range(10):
    with open(OWT_FILE_PATH, 'rb') as f:
        owt_samples = [sample_doc(f, ST.encode('utf-8')) for _ in range(10)]
    with open(TS_FILE_PATH, 'rb') as f:
        ts_samples = [sample_doc(f, ST.encode('utf-8')) for _ in range(10)]
    ts_result.append(compression_ratio(ts_tokenizer, ts_samples))
    owt_result.append(compression_ratio(owt_tokenizer, owt_samples))

print("TS Result ====")
print(ts_result)
print("max: ", max(ts_result))
print("min: ", min(ts_result))
print("median: ", stats.median(ts_result))
print("std: ", stats.stdev(ts_result))
print("OWT Result ====")
print(owt_result)
print("max: ", max(owt_result))
print("min: ", min(owt_result))
print("median: ", stats.median(owt_result))
print("std: ", stats.stdev(owt_result))
