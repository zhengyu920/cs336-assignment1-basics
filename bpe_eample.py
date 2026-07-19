from typing import Tuple


def bytes_to_tuple(input: bytes) -> tuple[bytes]:
    result = []
    for i in range(len(input)):
        result.append(input[i: i+1])
    print(result)
    return tuple(result)


def preprocess(content: str) -> dict[tuple[bytes], int]:
    w_counts = {}
    for w in content.split():
        w_encode = bytes_to_tuple(w.encode('utf-8'))
        w_counts[w_encode] = w_counts.get(w_encode, 0) + 1
    print('Preprocess result: \n', w_counts)
    return w_counts


def find_max_bp(w_counts: dict) -> tuple[bytes, int]:
    bp_counter = {}
    for w, count in w_counts.items():
        for i in range(len(w) - 1):
            bp = w[i] + w[i+1]
            bp_counter[bp] = bp_counter.get(bp, 0) + count

    bp_max = b''
    bp_max_count = 0
    for bp, count in bp_counter.items():
        if count > bp_max_count:
            bp_max = bp
            bp_max_count = count
        elif count == bp_max_count and bp > bp_max:
            bp_max = bp
            bp_max_count = count
    return bp_max, bp_max_count


def try_merge_bp(w: tuple[bytes], bp_to_merge: bytes) -> tuple[bytes]:
    if (len(w) < 2):
        return w

    new_w = []
    i = 0
    while i < len(w):
        if i < len(w) - 1 and w[i] + w[i+1] == bp_to_merge:
            new_w.append(bp_to_merge)
            i += 2
        else:
            new_w.append(w[i])
            i += 1
    return tuple(new_w)


def merge(w_counts: dict, bp_to_merge: bytes) -> dict[tuple[bytes], int]:
    result = {}
    for w, count in w_counts.items():
        new_w = try_merge_bp(w, bp_to_merge)
        result[new_w] = result.get(new_w, 0) + count
    return result


with open('data/bpe_example.txt') as f:
    content = f.read()
w_counts = preprocess(content)

new_vocab = []
for i in range(6):
    print('Iter: ', i)
    bp_to_merge, count = find_max_bp(w_counts)
    new_vocab.append(bp_to_merge)
    print('Count Result', bp_to_merge, count)
    w_counts = merge(w_counts, bp_to_merge)
    print('After Merge: \n', w_counts)
print("New vocab: ", new_vocab)
