"""Reproducible checkpoint versus random character-level GPT generation."""
import argparse
import json
import math
from pathlib import Path
import pickle
import sys
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'nanoGPT'))
from model import GPT, GPTConfig


@torch.no_grad()
def generate_text(model, ids, max_new_tokens=100, temperature=.8, top_k=200, seed=1337):
    if not ids:
        raise ValueError('prompt must not be empty')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if top_k is not None and top_k <= 0:
        raise ValueError('top_k must be positive')
    if max_new_tokens < 0:
        raise ValueError('max_new_tokens must be nonnegative')
    if any(i < 0 or i >= model.config.vocab_size for i in ids):
        raise ValueError('token id outside vocabulary')
    torch.manual_seed(seed)
    model.eval()
    x = torch.tensor(ids, dtype=torch.long, device=next(model.parameters()).device)[None, :]
    return model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)[0].tolist()


def load_checkpoint(path, device='cpu'):
    # Load only trusted local nanoGPT checkpoints: these include Python metadata.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = GPT(GPTConfig(**checkpoint['model_args']))
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()})
    return model.to(device).eval(), checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'nanoGPT/out-shakespeare-char-cpu/ckpt.pt')
    p.add_argument('--meta', type=Path, default=ROOT / 'nanoGPT/data/shakespeare_char/meta.pkl')
    p.add_argument('--prompt', default='\n')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--num-samples', type=int, default=2)
    p.add_argument('--max-new-tokens', type=int, default=300)
    p.add_argument('--temperature', type=float, default=.8)
    p.add_argument('--top-k', type=int, default=200)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    try:
        if a.num_samples <= 0 or a.threads <= 0:
            raise ValueError('num-samples and threads must be positive')
        if not a.checkpoint.is_file() or not a.meta.is_file():
            raise ValueError('checkpoint or meta file missing')
        torch.set_num_threads(a.threads)
        with a.meta.open('rb') as f:
            meta = pickle.load(f)
        try:
            ids = [meta['stoi'][c] for c in a.prompt]
        except KeyError as e:
            raise ValueError(f'unknown prompt character: {e}') from e
        trained, ckpt = load_checkpoint(a.checkpoint, a.device)
        if trained.config.vocab_size != meta['vocab_size']:
            raise ValueError('checkpoint/meta vocabulary mismatch')
        torch.manual_seed(a.seed)
        untrained = GPT(GPTConfig(**ckpt['model_args'])).to(a.device)
        result = {'checkpoint': str(a.checkpoint.resolve()), 'iter_num': ckpt['iter_num'],
                  'best_val_loss': float(ckpt['best_val_loss']), 'model_args': ckpt['model_args'],
                  'sampling': {'seed': a.seed, 'temperature': a.temperature, 'top_k': a.top_k,
                               'max_new_tokens': a.max_new_tokens, 'prompt': a.prompt}, 'samples': {}}
        for name, model in [('trained', trained), ('untrained', untrained)]:
            result['samples'][name] = []
            for i in range(a.num_samples):
                tokens = generate_text(model, ids, a.max_new_tokens, a.temperature, a.top_k, a.seed+i)
                text = ''.join(meta['itos'][t] for t in tokens)
                result['samples'][name].append(text)
                print(f'\n{name.upper()} sample {i+1}\n{text}')
        if a.output:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    except (ValueError, OSError, RuntimeError, KeyError) as e:
        p.exit(2, f'error: {e}\n')


if __name__ == '__main__':
    main()
