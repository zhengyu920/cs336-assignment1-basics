from pathlib import Path
import regex as re

num_iter: int = 6
vocab = [i for i in range(256)]
merge_list = []
inverted_merge_list = {}
next_id: int = 256


def find_max_bp(pretoken_count: dict):
    # Count byte pairs
    bp_counter = {}
    for encoded_tuple, count in pretoken_count.items():
        for i in range(len(encoded_tuple) - 1):
            bp = (encoded_tuple[i], encoded_tuple[i+1])
            bp_counter[bp] = bp_counter.get(bp, 0) + count
    max_bp, max_count = (0, 0), 0
    for bp, bp_count in bp_counter.items():
        if bp_count > max_count:
            max_bp = bp
            max_count = bp_count
        if bp_count == max_count and bp > max_bp:
            max_bp = bp
            max_count = bp_count
    return max_bp, max_count


def transform_and_count_pretoken(pretoken: list) -> dict:
    pretoken_count = {}
    for pt in pretoken:
        encoded_tuple = tuple(pt.encode('utf-8'))
        pretoken_count[encoded_tuple] = pretoken_count.get(
            encoded_tuple, 0) + 1
    return pretoken_count


def merge_bp(pretoken: dict, bp_merge: tuple):
    new_pretoken = {}
    origin, target = bp_merge
    for pt, count in pretoken.items():
        new_pt = []
        i = 0
        while i < len(pt):
            if i < len(pt) - 1 and (pt[i], pt[i+1]) == origin:
                new_pt.append(target)
                i = i + 2
            else:
                new_pt.append(pt[i])
                i = i + 1
        new_pretoken[tuple(new_pt)] = count
    return new_pretoken


def decode(tokens: tuple) -> str:
    result = ""
    for t in tokens:
        if t < 256:
            result = result + chr(t)
        else:
            original_bp = inverted_merge_list[t]
            result = result + decode(original_bp)
    return result


def pretokenizer() -> dict[list[bytes], int]:
    return {}


def train(file: str):
    global next_id
    global num_iter
    content = Path(file).read_text()
    pretoken = re.split(r'[\s\n]+', content)
    print(pretoken)
    pretoken_count = transform_and_count_pretoken(
        pretoken)  # {tuple(bytes) -> int}
    print(pretoken_count)
    for _ in range(num_iter):
        max_bp, max_count = find_max_bp(pretoken_count)
        if max_count <= 1:
            return   # end the training if not bp count greater than 1
        vocab.append(next_id)
        merge_list.append((max_bp, next_id))
        inverted_merge_list[next_id] = max_bp
        print(decode(max_bp), max_count)
        pretoken_count = merge_bp(pretoken_count, (max_bp, next_id))
        print(pretoken_count)
        next_id = next_id + 1

    p = []
    for w in vocab:
        c = (w,)
        p.append(decode(c))
    print(p)


if __name__ == "__main__":
    train('./cs336_basics/bpe_example.txt')
