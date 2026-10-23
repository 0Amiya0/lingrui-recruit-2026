"""Compare flash (scaled_dot_product_attention) vs manual attention speed on CPU."""
import os, sys, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nanoGPT'))
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(0)
B, T, C, n_head = 64, 256, 128, 4
hs = C // n_head
x = torch.randn(B, T, C)
q = torch.randn(B, n_head, T, hs)
k = torch.randn(B, n_head, T, hs)
v = torch.randn(B, n_head, T, hs)

def manual():
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
    mask = torch.tril(torch.ones(T, T)).view(1,1,T,T)
    att = att.masked_fill(mask == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    return att @ v

def flash():
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

for name, fn in [('manual', manual), ('flash/sdpa', flash)]:
    fn()  # warmup
    t0 = time.time()
    for _ in range(10):
        fn()
    print(f"{name}: {(time.time()-t0)/10*1000:.1f} ms/attn")
