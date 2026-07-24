# Problem (unicode1): Understanding Unicode (1 point)

(a) What Unicode character does chr(0) return?

`'\x00'`

---

(b) How does this character’s string representation (__repr__()) differ from its printed representation?

```
>>> chr(0).__repr__()
"'\\x00'"
>>> print(chr(0))

>>>
```
printed representation is nothing/empty.

---

(c) What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:
```
>>> chr(0).__repr__()
"'\\x00'"
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```
it is a null charactor that won't get print out.

# Problem (unicode2): Unicode Encodings (3 points)

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.
> Deliverable: A one-to-two sentence response.

UTA-8 represents the same string with much less bytes compared to others, so it is much more efficent.

---

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string
into a Unicode string. Why is this function incorrect? Provide an example of an input byte
string that yields incorrect results.
```
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
  return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
> Deliverable: An example input byte string for which `decode_utf8_bytes_to_str_wrong`
produces incorrect output, with a one-sentence explanation of why the function is incorrect.

UTF-8 is variable-width encoding, but that function treats UTF-8 as fixed-width of 1 byte.

For example:
```
>>> decode_utf8_bytes_to_str_wrong("こんにちは!".encode("utf-8"))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
  File "<stdin>", line 2, in <listcomp>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe3 in position 0: unexpected end of data
```

---

(c) Give a two-byte sequence that does not decode to any Unicode character(s).
> Deliverable: An example, with a one-sentence explanation.

```
>>> b = bytes([0x80, 0x80])
>>> b.decode('utf-8')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

This byte pair doesn't follow the utf-8 encoding schema.

---

# Problem (train_bpe_tinystories):  BPE Training on TinyStories (2 points)

(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary
size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the
vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How
much time and memory did training take? 

What is the longest token in the vocabulary? Does it make sense?

Resource requirements: ≤ 30 minutes (no GPUs), ≤ 30 GB RAM

Hint  You should be able to get under 2 minutes for BPE training using multiprocessing
during pre-tokenization and the following two facts:
- (a) The <|endoftext|> token delimits documents in the data files.
- (b) The <|endoftext|> token is handled as a special case before the BPE merges are applied.

Deliverable: A one-to-two sentence response.

It took less than 12GB memory and 53 seconds on my desktop.

Longest token:  b' accomplishment'
Length:  15

The word is indeed long and common so it makes sense.

---

(b) Profile your code. What part of the tokenizer training process takes the most time?

Deliverable: A one-to-two sentence response.

It is pretokeninzation. It takes over 50s with 4 workers and my entire training is under 1min.

# Problem (train_bpe_expts_owt):  BPE Training on OpenWebText (2 points)


(a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary
size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection.
What is the longest token in the vocabulary? Does it make sense?

Resource requirements: ≤ 12 hours (no GPUs), ≤ 100 GB RAM

Deliverable: A one-to-two sentence response.

Longest token:  ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ
Length:  64

It doesn't really make sense since ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ doesn't not like a real word.

---

(b) Compare and contrast the tokenizer that you get training on TinyStories versus
OpenWebText.

Deliverable: A one-to-two sentence response.

A lot of words in TinyStories vocab doesn't exists in OWT vocab, also comparing the longest token only the TinyStories is a real word. That probably means TinyStories has much higher data quality than OWT.


# Problem (tokenizer_experiments):  Experiments with tokenizers (4 points)

(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained
TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively),
encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio
(bytes/token)?

Deliverable: A one-to-two sentence response.

Run the 10 sample experiment 10 times:
tinystories compression ratio is about 4.0;
OpenWebText compression ratio is about 4.4;

---

(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer?
Compare the compression ratio and/or qualitatively describe what happens.

Deliverable: A one-to-two sentence response.

compression ratio drops to ~3.2 from ~4.0 after a couple runs.

ts tokenizer is train only on ts dataset so when it applies to owt dataset, it might a lot new words which will translate into single bytes.

---

(c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to
tokenize the Pile dataset (825GB of text)?

Deliverable: A one-to-two sentence response.

---

(d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and
development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype
uint16. Why is uint16 an appropriate choice?
Deliverable: A one-to-two sentence response.


