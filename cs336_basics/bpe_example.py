from pathlib import Path
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT = '<|endoftext|>'

def init_bpe():
    # dict[bytes, num]
    bpe = {EOT.encode('utf-8') : 0}
    for i in range(256):
        b_obj = bytes([i])
        bpe[b_obj] = 0
    return bpe

def train_step(bpe, pre_token_count):
    for pre_token, count in pre_token_count.items():
        utf_encoded = pre_token.encode('utf-8')
        for bp in bpe.keys():
            if bp in utf_encoded:
                bpe[bp] = bpe[bp] + count

def train(file : str):
    content = Path(file).read_text()
    pre_token = re.split(r'[\s\n]+', content)
    print(pre_token)
    pre_token_count = {}
    for w in pre_token:
        pre_token_count[w] = pre_token_count.get(w, 0) + 1
    print(pre_token_count)
    
    bpe = init_bpe()
    train_step(bpe, pre_token_count)
    print(bpe)

    

if __name__ == "__main__":
    print(Path.cwd())
    train('./cs336_basics/bpe_example.txt')