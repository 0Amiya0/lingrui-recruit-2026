"""Measure current environment and verify existing local character encoding."""
import datetime
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from generate_demo import load_checkpoint
torch.set_num_threads(1)
model, checkpoint = load_checkpoint(ROOT/'nanoGPT/out-shakespeare-char-cpu/ckpt.pt')
data = ROOT/'nanoGPT/data/shakespeare_char'
text = (data/'input.txt').read_text(encoding='utf-8')
with (data/'meta.pkl').open('rb') as file:
    meta = pickle.load(file)
encoded = np.array([meta['stoi'][char] for char in text], dtype=np.uint16)
split = int(len(text)*0.9)
result = {}
for name, expected in [('train', encoded[:split]), ('val', encoded[split:])]:
    actual = np.fromfile(data/(name+'.bin'), dtype=np.uint16)
    assert np.array_equal(actual, expected), name+' encoding mismatch'
    result[name] = {'tokens': int(actual.size), 'bytes': int(actual.nbytes), 'matches_local_raw_text_encoding': True}
assert all(meta['itos'][v] == k for k,v in meta['stoi'].items())
info = {'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'processor': platform.processor(), 'torch': torch.__version__, 'numpy': np.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_device_count': torch.cuda.device_count(), 'torch_threads': torch.get_num_threads(), 'nanoGPT_revision': subprocess.check_output(['git', '-c', f'safe.directory={ROOT / "nanoGPT"}', 'rev-parse', 'HEAD'], cwd=ROOT/'nanoGPT', text=True).strip(), 'total_parameters': sum(p.numel() for p in model.parameters()), 'parameters_excluding_position_embeddings': model.get_num_params(), 'checkpoint_iteration': checkpoint['iter_num'], 'best_val_loss': float(checkpoint['best_val_loss']), 'model_args': checkpoint['model_args'], 'checkpoint_config': checkpoint['config'], 'vocab_size': meta['vocab_size'], 'raw_text_characters': len(text), 'data': result}
target = Path(__file__).with_name('environment.json')
target.write_text(json.dumps(info, indent=2), encoding='utf-8')
print(json.dumps(info, indent=2))
