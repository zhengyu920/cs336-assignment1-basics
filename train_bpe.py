import os
from typing import BinaryIO
from pretokenization import pretokenize
from multiprocessing import Pool


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token,
                      bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def bytes_to_tuple(input: bytes) -> tuple[bytes]:
    result = []
    for i in range(len(input)):
        result.append(input[i: i+1])
    return tuple(result)


def process_chunk(input_path: str | os.PathLike,
                  special_tokens: list[str],
                  start_pos: int,
                  end_pos: int) -> dict[str, int]:
    with open(input_path, "rb") as f:
        f.seek(start_pos)
        chunk = f.read(end_pos - start_pos).decode("utf-8", errors="ignore")
        pretoken = pretokenize(chunk, special_tokens)
        counter = {}
        for token in pretoken:
            counter[token] = counter.get(token, 0) + 1
        return counter


def init_w_counter(input_path: str | os.PathLike,
                   special_tokens: list[str],
                   num_processes: int = 4
                   ) -> dict[tuple[bytes], int]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # prepare subprocess args
        args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            args.append((input_path, special_tokens, start, end))
        with Pool(num_processes) as p:
            all_counters = p.starmap(process_chunk, args)
    counter = {}
    for c in all_counters:
        for token, count in c.items():
            counter[token] = counter.get(token, 0) + count
    bytes_counter = {}
    for k, v in counter.items():
        bytes_tuple = bytes_to_tuple(k.encode('utf-8'))
        bytes_counter[bytes_tuple] = v
    return bytes_counter


def find_max_bp(bp_counter
                ) -> tuple[tuple[bytes, bytes] | None, int]:
    if len(bp_counter) == 0:
        return None, 0

    bp_max = None
    bp_max_count = 0
    for bp, count in bp_counter.items():
        if count > bp_max_count:
            bp_max = bp
            bp_max_count = count
        elif count == bp_max_count and bp > bp_max:
            bp_max = bp
            bp_max_count = count
    return bp_max, bp_max_count


def try_merge_bp(w: tuple[bytes],
                 bp_to_merge: tuple[bytes, bytes]
                 ) -> tuple[bytes]:
    if (len(w) < 2):
        return w

    merged_bp = bp_to_merge[0] + bp_to_merge[1]

    new_w = []
    i = 0
    while i < len(w):
        if i < len(w) - 1 and w[i] + w[i+1] == merged_bp:
            new_w.append(merged_bp)
            i += 2
        else:
            new_w.append(w[i])
            i += 1
    return tuple(new_w)


def merge(w_counter: dict,
          bp_counter: dict,
          bp_w_map: dict,
          w_bp_map: dict,
          bp_to_merge: tuple[bytes, bytes],
          ):
    w_to_merge = bp_w_map[bp_to_merge]

    for old_w in w_to_merge:
        new_w = try_merge_bp(old_w, bp_to_merge)
        w_count = w_counter[old_w]
        # update w_counter
        w_counter[new_w] = w_counter.get(new_w, 0) + w_count
        w_counter.pop(old_w)
        # update bp_counter and bp_w_map
        # deduct all bp in w from bp_counter and bp_w_map
        for bp in w_bp_map[old_w]:
            if bp != bp_to_merge:
                bp_counter[bp] -= w_count
                bp_w_map[bp].discard(old_w)
        # update w_bp_map
        # revmoe from w_bp_map
        w_bp_map.pop(old_w)
        # add new_w bps
        for i in range(len(new_w) - 1):
            bp = (new_w[i], new_w[i+1])
            bp_counter[bp] = bp_counter.get(bp, 0) + w_count
            bp_w_map.setdefault(bp, set()).add(new_w)
            w_bp_map.setdefault(new_w, []).append(bp)
    bp_counter.pop(bp_to_merge)
    bp_w_map.pop(bp_to_merge)


def init_bp_count(w_counter: dict[tuple[bytes], int]
                  ):
    bp_counter: dict[tuple[bytes, bytes], int] = {}
    bp_w_map: dict[tuple[bytes, bytes], set[tuple[bytes]]] = {}
    w_bp_map: dict[tuple[bytes], list[tuple[bytes, bytes]]] = {}
    for w, count in w_counter.items():
        for i in range(len(w) - 1):
            bp = (w[i], w[i+1])
            bp_counter[bp] = bp_counter.get(bp, 0) + count
            bp_w_map.setdefault(bp, set()).add(w)
            w_bp_map.setdefault(w, []).append(bp)

    return bp_counter, bp_w_map, w_bp_map


def train_bpe(input_path: str | os.PathLike,
              vocab_size: int,
              special_tokens: list[str],
              num_processes: int = 4
              ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {}
    merges = []
    for i in range(256):
        idx = len(vocab)
        vocab[idx] = i.to_bytes()
    for st in special_tokens:
        idx = len(vocab)
        vocab[idx] = st.encode('utf-8')
    # print('Initial vocab: \n', vocab)

    # pre-tokenization
    w_counter = init_w_counter(input_path, special_tokens, num_processes)

    # train bpe
    bp_counter, bp_w_map, w_bp_map = init_bp_count(w_counter)
    while len(vocab) < vocab_size:
        bp_to_merge, _ = find_max_bp(bp_counter)
        if bp_to_merge is None:
            break
        merge(w_counter, bp_counter, bp_w_map, w_bp_map, bp_to_merge)
        vocab[len(vocab)] = bp_to_merge[0] + bp_to_merge[1]
        merges.append(bp_to_merge)
    return (vocab, merges)


if __name__ == '__main__':
    path = 'data/TinyStoriesV2-GPT4-valid.txt'
    # path = 'data/bpe_example.txt'
    special_tokens = ['<|endoftext|>']
    vocab, merges = train_bpe(path, 500, special_tokens)
    print("Vocab: \n", vocab)
    print("Merges: \n", merges)
