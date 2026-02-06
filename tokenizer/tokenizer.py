from pathlib import Path
import pickle
import regex as re
from typing import Iterable
from typing import Iterator

class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    invalid_data = b'\xff'

    def __init__(self, vocab : dict[int, bytes], merges : list[tuple[bytes, bytes]], special_tokens :  list[str] | None=None):
        self._vocab = vocab
        self._vocab_size = len(vocab)
        self._inverted_vocab = {v: k for k, v in self._vocab.items()}
        self._merges = merges
        self._merges_size = len(merges)

        if special_tokens is None:
            self._rm_st_pattern = None
            self._special_tokens = None
        else:
            sp = special_tokens
            sp.sort(key=len, reverse=True)
            self._rm_st_pattern = f"({'|'.join(map(re.escape, sp))})"
            self._special_tokens = set(sp)
            for st in sp:
                st_encoded = st.encode('utf-8')
                if st_encoded in self._inverted_vocab:
                    continue
                self._vocab[self._vocab_size] = st_encoded
                self._inverted_vocab[st_encoded] = self._vocab_size
                self._vocab_size = self._vocab_size + 1
        

    def from_files(cls, vocab_filepath = str, merges_filepath = str, special_tokens : list[str] | None=None):
        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_path, 'rb') as f:
            merges = pickle.load(f)

        return Tokenizer(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        words = self._pretokenization(text)
        encoded_output = []
        for w in words:
            if isinstance(w, str):
                encoded_output.append(self._inverted_vocab[w.encode('utf-8')])
                continue
            merged_word = self._merge_word(w)
            for sub_word_bytes in merged_word:
                encoded_output.append(self._inverted_vocab[sub_word_bytes])
        return encoded_output
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            encoded = self.encode(text)
            for e in encoded:
                yield e

    def decode(self, ids: list[int]) -> str:
        builder = bytearray()
        for id in ids:
            builder.extend(self._vocab.get(id, Tokenizer.invalid_data))
        return bytes(builder).decode('utf-8', errors='replace')
    
    def _pretokenization(self, text: str):
        st_processed_text = self._process_special_tokens(text)
        return self._compile_to_bytes_list(st_processed_text)
    
    def _process_special_tokens(self, content: str) -> list[str]:
        if self._rm_st_pattern is None:
            return [content]
        return re.split(self._rm_st_pattern, content)
    
    # turn each of the str in text into utf-8 encoded bytes
    def _compile_to_bytes_list(self, docs: list[str]) -> list[list[bytes]]:
        bytes_list = []
        for doc in docs:
            if self._special_tokens is not None and doc in self._special_tokens:
                bytes_list.append(doc)
                continue
            matches = re.finditer(Tokenizer.PAT, doc)
            for match in matches:
                word = list(bytes([b]) for b in match.group().encode('utf8'))
                bytes_list.append(word)
        return bytes_list
    
    def _merge_word(self, word: list[bytes]) -> list[bytes]:
        # for each merges, scan the word from left to right and merge them
        for merge in self._merges:
            word = self._merge_step(word, merge)
            if len(word) == 1:
                return word
        return word
    
    # perform merge for a single merge tuple
    def _merge_step(self, word: list[bytes], merge: tuple[bytes, bytes]) -> list[bytes]:
        merged_word = []
        merged = merge[0] + merge[1]
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == merge:
                merged_word.append(merged)
                i = i + 2
            else:
                merged_word.append(word[i])
                i = i + 1
        return merged_word

if __name__ == '__main__':
    vocab = {0: b' ', 1: b'a', 2:b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    st = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    t = Tokenizer(vocab, merges, st)
    text = 'the cat<|endoftext|><|endoftext|> ate<|endoftext|>'
    print(t._process_special_tokens(text))
    print(t._pretokenization(text))
    encoded = t.encode(text)
    print(encoded)
    decoded = t.decode(encoded)
    print(decoded)

    # test  = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    # print(test.encode('utf-8'))
    # print(t._pretokenization(test))
    # print(t.encode(test))
    # print(t.decode(t.encode(test)))