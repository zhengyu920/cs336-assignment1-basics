import regex as re

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


@profile
def pretokenize(input_text: str, special_tokens: list[str]) -> list[str]:
    splits = []
    if len(special_tokens) == 0:
        splits = [input_text]
    else:
        split_pattern = '|'.join([re.escape(st) for st in special_tokens])
        splits = re.split(split_pattern, input_text)
    result = []
    for split in splits:
        for match in re.finditer(PAT, split):
            result.append(match.group())
    return result


if __name__ == '__main__':
    st = ['<|endoftext|>', '<eos>']
    print(pretokenize("some text that i'll pre-tokenize<|endoftext|>", st))
    print(pretokenize("<eos><eos>", st))
    print(pretokenize("", st))
    print(pretokenize("some text that i'll pre-tokenize<|endoftext|>", []))
