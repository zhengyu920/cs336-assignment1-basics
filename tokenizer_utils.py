def bytes_to_tuple(input: bytes) -> tuple[bytes]:
    result = []
    for i in range(len(input)):
        result.append(input[i: i+1])
    return tuple(result)