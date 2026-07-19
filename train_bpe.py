import os
from typing import BinaryIO

from pretokenization import pretokenize


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


def init_counter(input_path: str | os.PathLike,
                 special_tokens: list[str]
                 ) -> dict[tuple[bytes], int]:
    counter = {}

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            pretokens = pretokenize(chunk, special_tokens)
            for pretoken in pretokens:
                bytes_tuple = bytes_to_tuple(pretoken.encode('utf-8'))
                counter[bytes_tuple] = counter.get(bytes_tuple, 0) + 1
    return counter


def find_max_bp(w_counts: dict[tuple[bytes], int]
                ) -> tuple[tuple[bytes, bytes] | None, int]:
    bp_counter = {}
    for w, count in w_counts.items():
        for i in range(len(w) - 1):
            bp = (w[i], w[i+1])
            bp_counter[bp] = bp_counter.get(bp, 0) + count

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


def try_merge_bp(w: tuple[bytes], bp_to_merge: tuple[bytes, bytes]) -> tuple[bytes]:
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


def merge(w_counts: dict, bp_to_merge: tuple[bytes, bytes]) -> dict[tuple[bytes], int]:
    result = {}
    for w, count in w_counts.items():
        new_w = try_merge_bp(w, bp_to_merge)
        result[new_w] = result.get(new_w, 0) + count
    return result


def train_bpe(input_path: str | os.PathLike,
              vocab_size: int,
              special_tokens: list[str]
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

    counter = init_counter(input_path, special_tokens)
    while len(vocab) < vocab_size:
        # print('Premerge: \n', counter)
        bp, _ = find_max_bp(counter)
        if bp is None:
            break
        # print('Bp to merge: ', bp, "Count: ", count)
        counter = merge(counter, bp)
        vocab[len(vocab)] = bp[0] + bp[1]
        merges.append(bp)
    return (vocab, merges)


if __name__ == '__main__':
    path = 'data/TinyStoriesV2-GPT4-valid.txt'
    # path = 'data/bpe_example.txt'
    special_tokens = ['<|endoftext|>']
    train_bpe(path, 259, special_tokens)
