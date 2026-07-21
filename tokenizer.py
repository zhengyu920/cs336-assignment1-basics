from typing import Iterable, Iterator

class Tokenizer:
    """
    Given a vocabulary and a list of merges, encodes
    text into integer IDs and decodes integer IDs into text.
    """

    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        return
    
    def from_files(cls, 
                   vocab_filepath: str, 
                   merges_filepath:str, 
                   special_tokens:list[str] | None = None):
        raise("not implemented")
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        raise("not implemented")
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files 
        that we cannot directly load into
        memory.
        """
        raise("not implemented")
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        raise("not implemented")