import os
import random
import io
from typing import BinaryIO


def sample_doc(
    file: BinaryIO,
    split_special_token: bytes,
) -> tuple[int, int]:
    """
    Sample a document from file. A document is delimited by split_special_token.
    """
    assert isinstance(split_special_token,
                      bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    initial_position = random.randint(0, file_size - 1)
    cur_posistion = initial_position

    start_pos = 0
    end_pos = 0

    # starting from init post, seek forward until it hits the first st or end of file.
    file.seek(cur_posistion)
    while True:
        mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

        # If EOF, this boundary should be at the end of the file
        if mini_chunk == b"":
            end_pos = file_size
            break

        # Find the special token in the mini chunk
        found_at = mini_chunk.find(split_special_token)
        if found_at != -1:
            end_pos = cur_posistion + found_at
            break
        cur_posistion += mini_chunk_size

    # starting from init post, seek backward until it hits the first st or start of file.
    cur_posistion = initial_position
    file.seek(max(0, cur_posistion - mini_chunk_size))
    while cur_posistion > 0:
        if (cur_posistion < mini_chunk_size):
            file.seek(0)
            chunk_size = cur_posistion
        else:
            file.seek(cur_posistion - mini_chunk_size)
            chunk_size = mini_chunk_size

        seek_pos = file.tell()
        mini_chunk = file.read(chunk_size)  # Read a mini chunk

        # Find the special token in the mini chunk
        found_at = mini_chunk.rfind(split_special_token)
        if found_at != -1:
            start_pos = seek_pos + found_at + len(split_special_token)
            break
        cur_posistion -= chunk_size

    # in case inital rand seek land in the middle of special token.
    # there will be a spcial token in the middle of the final chunk.
    file.seek(start_pos)
    chunk = fake_file.read(end_pos - start_pos)
    found_at = chunk.find(split_special_token)
    if found_at != -1:
        end_pos = start_pos + found_at

    return (start_pos, end_pos)


if __name__ == '__main__' :
    fake_file = io.BytesIO(b"doc1<|endoftext|>doc2<|endoftext|>doc3")
    st = b"<|endoftext|>"
    start_pos, end_pos = sample_doc(fake_file, st)
    fake_file.seek(start_pos)
    chunk = fake_file.read(end_pos - start_pos).decode("utf-8", errors="ignore")
    print(chunk)