from pathlib import Path
import pickle
class Tokenizer:
    def __init__(self, vocab : dict[int, bytes], merges : list[tuple[bytes, bytes]], special_tokens :  list[str] | None=None):
        self._vocab = vocab
        self._merges = merges
        self._special_tokens = special_tokens

    def from_files(cls, vocab_filepath = str, merges_filepath = str, special_tokens : list[str] | None=None):
        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_path, 'rb') as f:
            merges = pickle.load(f)

        return Tokenizer(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        return []
    
    def decode(self, ids: list[int]) -> str:
        return ""