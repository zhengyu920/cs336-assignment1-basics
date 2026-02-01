## Problem (unicode1): Understanding Unicode (1 point)

**(a) What Unicode character does chr(0) return?**

`'\x00'`

**(b) How does this character’s string representation (`__repr__()`) differ from its printed representation?**

`chr(0).__repr__()` shows `'\x00'`, and `print(chr(0))` shows nothing. It is a null character.

**(c) What happens when this character occurs in text?**

It may be helpful to playaround with the following in your Python interpreter and see if it matches your expectations:

```
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

In string representation, it is null character itself `'this is a test\x00string'`. In printed representation, it is nothing `this is a teststring`.


## Problem (unicode2): Unicode Encodings (3 points)

**(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.**

Most of the time, 1 byte is one character.

**(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.**

```
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

Some utf8 needs more than 1 bytes to represent.

**(c) Give a two byte sequence that does not decode to any Unicode character(s).**


