def concat_merge(merge):
    part1 = bytes([merge[0]]) if isinstance(merge[0], int) else merge[0]
    part2 = bytes([merge[1]]) if isinstance(merge[1], int) else merge[1]
    return part1 + part2

def to_bytes_tuple(merge):
    part1 = bytes([merge[0]]) if isinstance(merge[0], int) else merge[0]
    part2 = bytes([merge[1]]) if isinstance(merge[1], int) else merge[1]
    return (part1, part2)