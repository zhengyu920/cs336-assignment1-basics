import regex as re

if __name__ == "__main__":
    print("hello world!")
    test_string = "hello! こんにちは!"
    utf8_encoded = test_string.encode('utf-8')
    print(utf8_encoded)
    print(type(utf8_encoded))
    print(list(utf8_encoded))
    print(utf8_encoded[2])
    print(utf8_encoded.decode('utf-8'))

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    print("some text that i'll pre-tokenize")
    print(re.findall(PAT, "some text that i'll pre-tokenize"))