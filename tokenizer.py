from typing import Iterable, Iterator
from pretokenization import pretokenize
from tokenizer_utils import bytes_to_tuple
import pickle


def merge_bytes(bs: tuple[bytes], merge: tuple[bytes, bytes]) -> tuple[bytes]:
        if len(bs) < 2:
            return bs
        new_bs = []
        idx = 0
        while idx < len(bs):
            if idx < len(bs) - 1 and (bs[idx], bs[idx+1]) == merge:
                new_bs.append(bs[idx]+bs[idx+1])
                idx += 2
            else:
                new_bs.append(bs[idx])
                idx += 1
        return tuple(new_bs)
class Tokenizer:
    """
    Given a vocabulary and a list of merges, encodes
    text into integer IDs and decodes integer IDs into text.
    """

    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_to_int = {}
        for k, v in vocab.items():
            if v in self.vocab_to_int:
                raise ValueError("Duplicated token in vocab")
            self.vocab_to_int[v] = k
    
    @classmethod
    def from_files(cls, 
                   vocab_filepath: str, 
                   merges_filepath:str, 
                   special_tokens:list[str] | None = None):
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary
        and list of merges (in the same format that your BPE training code output) 
        and (optionally) a list of special tokens.
        """
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        pretokens = pretokenize(text, self.special_tokens)
        encoded = []
        for pretoken in pretokens:
            bs = bytes_to_tuple(pretoken)
            for merge in self.merges:
                bs = merge_bytes(bs, merge)
            encoded.extend([self.vocab_to_int[b] for b in bs])
        return encoded
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files 
        that we cannot directly load into
        memory.
        """
        raise NotImplementedError("encode_iterable not implemented")
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        return b"".join([self.vocab[id] for id in ids]).decode('utf-8' , errors='replace')