import os
from typing import BinaryIO
import regex as re
import multiprocessing
from pathlib import Path
from tokenizer.thread_safe_counter import ThreadSafeCounter

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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


def remove_special_tokens(content: str, special_tokens: list[str]) -> list[str]:
    pattern = '|'.join(re.escape(d) for d in special_tokens)
    return re.split(pattern, content)


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize_chunk(content: str, special_tokens: list[str], output_counter: ThreadSafeCounter):
    docs = remove_special_tokens(content, special_tokens)
    counter = {}
    for doc in docs:
        matches = re.finditer(PAT, doc)

        for match in matches:
            text = tuple(bytes([b]) for b in match.group().encode('utf8'))
            counter[text] = counter.get(text, 0) + 1
    output_counter.increment_from_dict(counter)

def process_with_multiprocessing(input_path: str | os.PathLike, special_tokens: list[str], num_processes: int):
    path = Path(input_path)
    with multiprocessing.Manager() as manager:
        counter = ThreadSafeCounter(manager)
        with open(path, "rb") as f:
            processes = []
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                p = multiprocessing.Process(target=pretokenize_chunk, args=(chunk, special_tokens, counter))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
        return counter.to_dict()

## Usage
def pretokenization(input_path: str | os.PathLike, special_tokens: list[str], num_processes: int = 8):
    return process_with_multiprocessing(input_path, special_tokens, num_processes)