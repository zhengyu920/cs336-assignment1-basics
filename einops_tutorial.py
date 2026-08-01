import torch
from einops import rearrange, reduce, repeat, einsum

import einops
print(einops.__version__)

def show(label, tensor:torch.Tensor):
    print(f"{label}: {tuple(tensor.shape)}")
    print(tensor)

x = torch.arange(24)
x = x.reshape(2,3,4)

show('x', x)

# out[b, d, s] = x[b, s, d]
# it is all about index change
# out[0, 0, 0] = x[0, 0, 0]
# out[0, 0, 1] = x[0, 1, 0]
# out[0, 0, 2] = x[0, 2, 0]
x1 = rearrange(x, 'b s d -> b d s')

show('b d s', x1)

# 'b s d -> b (s d)'
# Output has 2 axes. The second one has size 3 × 4 = 12, and its index is built from s and d by:
# flat = s * 4 + d
# That's the leftmost-varies-slowest rule
show('b (s d)', rearrange(x, 'b s d -> b (s d)'))
show('b (d s)', rearrange(x, 'b s d -> b (d s)'))

print(rearrange(x, 'b s d -> b d s').untyped_storage().data_ptr())
print(rearrange(x, 'b s d -> b (s d)').untyped_storage().data_ptr())
print(rearrange(x, 'b s d -> b (d s)').untyped_storage().data_ptr())

try:
    rearrange(x, 'b s (h dh) -> b h s dh', h=5)
except einops.EinopsError as e:
    print(e)

# use index arithmatic to figure out the data position
# d = h * 2 + dh
# d = dh * 2 + h
show('b s (h dh) -> b h s dh', rearrange(x, 'b s (h dh) -> b h s dh', h=2))
show('b s (dh h) -> b h s dh', rearrange(x, 'b s (dh h) -> b h s dh', h=2))


# Any axis on the left that is missing from the right gets reduced away.
show('b s d -> b s', reduce(x.float(), 'b s d -> b s', 'mean'))
show('hormalize', x - reduce(x.float(), 'b s d -> b s 1', 'mean'))
show('b s d -> s d', reduce(x, 'b s d -> s d', 'sum'))
show('b s d -> b d', reduce(x, 'b s d -> b d', 'max'))
show('reduce #5', reduce(x, 'b s (d1 d2) -> b s d1', 'max', d1=2))

show('repeat b s d -> b s d k', repeat(x, 'b s d -> b s d k', k=3))


# 1. Collect every distinct letter in the pattern.
# 2. Loop over ALL combinations of their index values.
# 3. For each combination: pull one element from each input (using that
#    input's own letters) and multiply them together.
# 4. Accumulate into the output slot named by the output letters.
#    Letters absent from the output get summed away in the process.
# Let a = [1, 2], b = [10, 20].

# 'i, i ->' — one distinct letter, i. Both inputs index by it, so they advance in lockstep:


# i=0:  a[0] * b[0] = 1*10 = 10
# i=1:  a[1] * b[1] = 2*20 = 40
# i is absent from the output → sum → 50. A scalar. Dot product.

# 'i, j -> i j' — two distinct letters. They vary independently, so the loop covers all four combinations:


# i=0,j=0:  a[0]*b[0] = 10
# i=0,j=1:  a[0]*b[1] = 20
# i=1,j=0:  a[1]*b[0] = 20
# i=1,j=1:  a[1]*b[1] = 40
# Both letters appear in the output → nothing is summed → each product lands in its own slot:


# [[10, 20],
#  [20, 40]]
a = torch.arange(6).reshape(2, 3).float()
b = torch.arange(12).reshape(3, 4).float()
show('#1', einsum(a ,b, 'i j, j k -> i k'))
assert torch.allclose(a@b, einsum(a ,b, 'i j, j k -> i k'))
show('#3', einsum(a, a, 'i j, i j -> i'))

def mean_over_last_axis(x):
    return reduce(x, '... d -> ...', 'mean')
show('#1 (4,)', mean_over_last_axis(torch.arange(4).float()))
show('#1 (2, 4)', mean_over_last_axis(torch.arange(8).reshape(2, 4).float()))
show('#1 (2, 4)', mean_over_last_axis(torch.arange(24).reshape(2, 3, 4).float()))