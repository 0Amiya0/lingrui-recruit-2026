"""Quick CPU timing benchmark for the nanoGPT model (standalone, no eval overhead)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nanoGPT'))
import torch
from model import GPTConfig, GPT

torch.manual_seed(1337)

def bench(n_layer, n_head, n_embd, block_size, batch_size, n_iters=5):
    cfg = GPTConfig(
        block_size=block_size, vocab_size=65, n_layer=n_layer,
        n_head=n_head, n_embd=n_embd, dropout=0.2, bias=False,
    )
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 65, (batch_size, block_size))
    y = torch.randint(0, 65, (batch_size, block_size))
    # warmup
    logits, loss = model(x, y); loss.backward(); opt.zero_grad()
    t0 = time.time()
    for _ in range(n_iters):
        logits, loss = model(x, y)
        loss.backward()
        opt.step(); opt.zero_grad()
    dt = (time.time() - t0) / n_iters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_layer={n_layer} n_head={n_head} n_embd={n_embd} block={block_size} batch={batch_size}: "
          f"{dt*1000:.0f} ms/iter  ({n_params/1e6:.2f}M params)")
    return dt

print(f"torch threads: {torch.get_num_threads()}")
bench(4, 4, 128, 256, 64)
bench(6, 6, 384, 256, 64)   # the original "baby GPT" for comparison
bench(4, 4, 128, 128, 64)
bench(4, 4, 128, 256, 32)
