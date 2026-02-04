import os
from tokenizer.pretokenization import pretokenization
import time
import statistics

def find_max_bp(pretokenized_count):
    # Count byte pairs
    bp_counter = {}
    for bytes_tuple, count in pretokenized_count.items():
        for i in range(len(bytes_tuple) - 1):
            bp = (bytes_tuple[i], bytes_tuple[i+1])
            bp_counter[bp] = bp_counter.get(bp, 0) + count
    max_bp, max_count = (b'', b''), 0
    for bp, bp_count in bp_counter.items():
        # print(max_bp, max_count, bp)
        if bp_count > max_count:
            max_bp = bp
            max_count = bp_count
        if bp_count == max_count and bp > max_bp:
            max_bp = bp
            max_count = bp_count
    return max_bp, max_count


def merge_bp(pretokenized_count: dict[tuple[bytes], int], merge):
    new_pretokenized_count = {}
    merged_bytes = merge[0] + merge[1]
    for pt, count in pretokenized_count.items():
        new_pt = []
        i = 0
        while i < len(pt):
            if i < len(pt) - 1 and (pt[i], pt[i+1]) == merge:
                new_pt.append(merged_bytes)
                i = i + 2
            else:
                new_pt.append(pt[i])
                i = i + 1
        new_pretokenized_count[tuple(new_pt)] = count
    return new_pretokenized_count

def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for st in special_tokens:
        idx = len(vocab)
        vocab[idx] = st.encode('utf8')
    return vocab

def print_perf_stats(pretokenization_time : float, find_map_bp_time : list[float], merge_bp_time: list[float]):
    fmb_time_sum = sum(find_map_bp_time)
    fmb_time_avg = statistics.mean(find_map_bp_time)
    fmb_time_median = statistics.median(find_map_bp_time)

    mb_time_sum = sum(merge_bp_time)
    mb_time_avg = statistics.mean(merge_bp_time)
    mb_time_median = statistics.median(merge_bp_time)

    print(f"Total execution time: {pretokenization_time + fmb_time_sum + mb_time_sum:.4f} seconds")
    print(f"Pretokenization execution time: {pretokenization_time:.4f} seconds")
    print(f"Find max BP time avg: {fmb_time_avg * 1000:.2f} ms")
    print(f"Find max BP time median: {fmb_time_median * 1000:.2f} ms")
    print(f"Merge BP time avg: {mb_time_avg * 1000:.2f} ms")
    print(f"Merge BP time median: {mb_time_median * 1000:.2f} ms")

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = init_vocab(special_tokens)
    merges = []
    start_time = time.perf_counter()
    print(f"Running pretokenizatoin with 8 processes...")
    pretokenized_count = pretokenization(input_path, special_tokens)
    print("Pretokenization is done...")
    end_time = time.perf_counter()
    pretokenization_time = end_time - start_time
    #print(f'prtokenization result: {pretokenized_count}')
    find_max_bp_time = []
    merge_bp_time = []
    print("Start training")
    while len(vocab) < vocab_size:
        start_time = time.perf_counter()
        max_bp, max_count = find_max_bp(pretokenized_count)
        end_time = time.perf_counter()
        find_max_bp_time.append(end_time - start_time)
        # only merge and update when max_count > 1.
        if max_count > 1:
            cur_size = len(vocab)
            vocab[cur_size] = max_bp[0] + max_bp[1]
            merges.append(max_bp)
            start_time = time.perf_counter()
            pretokenized_count = merge_bp(pretokenized_count, max_bp)
            end_time = time.perf_counter()
            merge_bp_time.append(end_time - start_time)
        else:
            break
    print("Training is done...")
    print_perf_stats(pretokenization_time, find_max_bp_time, merge_bp_time)
    return vocab, merges

import pickle

if __name__ == "__main__":
    input_path = './data/TinyStoriesV2-GPT4-valid.txt'
    # input_path = './data/TinyStoriesV2-GPT4-train.txt'
    # input_path = './tokenizer/bpe_example.txt'
    special_tokens = ['<|endoftext|>']
    vocab_size = 10000
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    with open("./output/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open('./output/merges.pkl', 'wb') as f:
        pickle.dump(merges, f)

    # print(vocab)
    # print(merges)