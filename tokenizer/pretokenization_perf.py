from tokenizer.pretokenization import pretokenization
import time

if __name__ == "__main__":
    # input_path = './data/TinyStoriesV2-GPT4-valid.txt'
    input_path = './data/TinyStoriesV2-GPT4-train.txt'
    # input_path = './tokenizer/bpe_example.txt'
    special_tokens = ['<|endoftext|>']
    test_set = [1, 2, 4, 8, 16, 32]
    for np in test_set:
        start = time.perf_counter()
        pretokenization(input_path, special_tokens, np)
        end = time.perf_counter()
        print(f"Num Processes: {np:3d} | Time: {(end - start):.4f} seconds")