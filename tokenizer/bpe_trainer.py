import os
from pathlib import Path
import regex as re


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


def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return {}, []
