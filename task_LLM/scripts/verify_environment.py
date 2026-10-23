"""Record measured environment and local checkpoint/data evidence."""
import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generate_demo import load_checkpoint


def main():
    torch.set_num_threads(4)
    model, ckpt = load_checkpoint(ROOT / 'nanoGPT/out-shakespeare-char-cpu/ckpt.pt')
    data = ROOT / 'nanoGPT/data/shakespeare_char'
    info = {'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'processor': platform.processor(), 'torch': torch.__version__, 'numpy': np.__version__,
            'cuda_available': torch.cuda.is_available(), 'torch_threads': torch.get_num_threads(),
            'nanoGPT_revision': subprocess.check_output(['git', '-c', f'safe.directory={ROOT / "nanoGPT"}', 'rev-parse', 'HEAD'], cwd=ROOT / 'nanoGPT', text=True).strip(),
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'parameters_excluding_position_embeddings': model.get_num_params(),
            'checkpoint_iteration': ckpt['iter_num'], 'best_val_loss': float(ckpt['best_val_loss']),
            'model_args': ckpt['model_args'],
            'data': {s: {'tokens': int(np.fromfile(data / f'{s}.bin', dtype=np.uint16).size),
                         'bytes': (data / f'{s}.bin').stat().st_size} for s in ('train', 'val')}}
    target = ROOT / 'artifacts/verification/environment.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
