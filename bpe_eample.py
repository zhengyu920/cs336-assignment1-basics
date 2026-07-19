def bytes_to_tuple(input: bytes):
    result = []
    for i in range(len(input)):
        result.append(input[i: i+1])
    print(result)
    return tuple(result)


def preprocess(content: str):
    w_count = {}
    for w in content.split():
        w_encode = bytes_to_tuple(w.encode('utf-8'))
        if (w_encode in w_count):
            w_count[w_encode] += 1
        else:
            w_count[w_encode] = 1
    print(w_count)


with open('data/bpe_example.txt') as f:
    content = f.read()
preprocess(content)
