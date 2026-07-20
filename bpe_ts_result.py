import pickle

with open('bpe_ts_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
    max_len = 0
    max_token = None
    for _, token in vocab.items():
        if len(token) > max_len:
            max_len = len(token)
            max_token = token
    print("Longest token: ", max_token)
    print("Length: ", max_len)

with open('bpe_ts_merges.pkl', 'rb') as f:
    merges = pickle.load(f)