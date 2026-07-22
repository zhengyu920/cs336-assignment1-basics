import pickle

with open('bpe_ts_vocab.pkl', 'rb') as f:
    vocab_ts = pickle.load(f)
    max_len_ts = 0
    max_token_ts = None
    for _, token in vocab_ts.items():
        if len(token) > max_len_ts:
            max_len_ts = len(token)
            max_token_ts = token
    print("Longest token TS: ", max_token_ts.decode('utf-8', errors='replace'))
    print("Length: ", max_len_ts)

with open('bpe_ts_merges.pkl', 'rb') as f:
    merges_ts = pickle.load(f)


with open('bpe_owt_vocab.pkl', 'rb') as f:
    vocab_owt = pickle.load(f)
    max_len_owt = 0
    max_token_owt = None
    for _, token in vocab_owt.items():
        if len(token) > max_len_owt:
            max_len_owt = len(token)
            max_token_owt = token
    print("Longest token OWT: ", max_token_owt.decode('utf-8', errors='replace'))
    print("Length: ", max_len_ts)

with open('bpe_ts_merges.pkl', 'rb') as f:
    merges_owt = pickle.load(f)

# TS has 10000 max vocab, and OWT has 32000 max vocab.
# so compare the first 9000 merges and for tokens in TS, if any of them not in OWT.
vocab_token_set_ts = set(vocab_ts.values())
vocab_token_set_owt = set(vocab_owt.values())

# for token in vocab_token_set_ts:
#     if token not in vocab_token_set_owt:
#         print(token.decode('utf-8', errors='replace'))