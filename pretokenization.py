import regex as re

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def pretokenize(input_text: str, special_tokens: list[str] | None = None) -> list[bytes]:
    if special_tokens is not None:
        special_tokens.sort(reverse=True)
    splits = []
    if special_tokens is None or len(special_tokens) == 0:
        splits = [input_text]
    else:
        split_pattern = '(' + '|'.join([re.escape(st) for st in special_tokens]) + ')'
        splits = re.split(split_pattern, input_text)
    result = []
    if special_tokens is not None:
        special_tokens = set(special_tokens)
    for split in splits:
        if special_tokens is not None and split in special_tokens:
            result.append(split.encode('utf-8'))
            continue
        for match in re.finditer(PAT, split):
            result.append(match.group().encode('utf-8'))
    return result

def pretokenize_into_counter(input_text: str, special_tokens: list[str]):
    special_tokens.sort(reverse=True)
    splits = []
    if len(special_tokens) == 0:
        splits = [input_text]
    else:
        split_pattern = '|'.join([re.escape(st) for st in special_tokens])
        splits = re.split(split_pattern, input_text)
    counter = {}
    for split in splits:
        for match in re.finditer(PAT, split):
            word = match.group()
            counter[word] = counter.get(word, 0) + 1
    return counter


if __name__ == '__main__':
    st = ['<|endoftext|>', '<eos>']
    print(pretokenize("some text that i'll pre-tokenize<|endoftext|>", st))
    print(pretokenize("<eos><eos>", st))
    print(pretokenize("", st))
    print(pretokenize("some text that i'll pre-tokenize<|endoftext|>", []))
    overlap_st = ['<eos>', '<eos><eos>']
    print(pretokenize("<eos><eos>", overlap_st))
