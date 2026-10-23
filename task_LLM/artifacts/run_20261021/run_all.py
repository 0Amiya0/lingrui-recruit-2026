"""Isolated local verification; leaves formal training and data untouched."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ENV = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
records = []
protected = [ROOT / p for p in ['nanoGPT/out-shakespeare-char-cpu/ckpt.pt', 'train_log.txt', 'loss_curve.png', 'generate_output.txt', 'nanoGPT/data/shakespeare_char/input.txt', 'nanoGPT/data/shakespeare_char/train.bin', 'nanoGPT/data/shakespeare_char/val.bin', 'nanoGPT/data/shakespeare_char/meta.pkl']]

def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}

def run(name, args, cwd=ROOT):
    started = time.time()
    with (OUT / (name + '.log')).open('w', encoding='utf-8') as log:
        completed = subprocess.run([sys.executable, *args], cwd=cwd, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    record = {'name': name, 'argv': [sys.executable, *args], 'cwd': str(cwd), 'returncode': completed.returncode, 'elapsed_seconds': round(time.time()-started, 3), 'log': name+'.log'}
    records.append(record)
    (OUT / 'commands.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)
    if completed.returncode:
        raise RuntimeError(f'{name} failed; inspect its log')

before = hashes()
(OUT / 'protected_before.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
run('tests', ['-c', "import torch,unittest; torch.set_num_threads(1); suite=unittest.defaultTestLoader.discover('tests'); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())"])
run('environment', [str(OUT / 'verify_local.py')])
generation = ['generate_demo.py', '--threads', '1', '--seed', '1337', '--num-samples', '2', '--max-new-tokens', '300']
run('generation', [*generation, '--output', str(OUT / 'generation.json')])
run('generation_repeat', [*generation, '--output', str(OUT / 'generation_repeat.json')])
assert json.loads((OUT/'generation.json').read_text(encoding='utf-8')) == json.loads((OUT/'generation_repeat.json').read_text(encoding='utf-8'))
run('historical_plot', ['plot_loss.py', '--log', 'train_log.txt', '--csv', str(OUT/'historical_loss.csv'), '--output', str(OUT/'historical_loss_curve.png')])
run('smoke_train', ['train.py', 'config/train_shakespeare_char_cpu.py', '--out_dir='+str(OUT/'smoke-model'), '--batch_size=2', '--block_size=32', '--n_layer=1', '--n_head=2', '--n_embd=32', '--max_iters=20', '--eval_interval=4', '--eval_iters=4', '--log_interval=1', '--warmup_iters=1', '--lr_decay_iters=20', '--always_save_checkpoint=True'], ROOT/'nanoGPT')
run('smoke_generation', ['generate_demo.py', '--threads', '1', '--checkpoint', str(OUT/'smoke-model/ckpt.pt'), '--num-samples', '1', '--max-new-tokens', '80', '--output', str(OUT/'smoke_generation.json')])
run('smoke_plot', ['plot_loss.py', '--log', str(OUT/'smoke_train.log'), '--csv', str(OUT/'smoke_loss.csv'), '--output', str(OUT/'smoke_loss_curve.png')])
after = hashes()
assert before == after, 'Historical artifacts changed'
(OUT/'verification.json').write_text(json.dumps({'protected_files_unchanged': before == after, 'generation_repeat_identical': True, 'protected_sha256': after, 'all_commands_exit_zero': all(r['returncode']==0 for r in records)}, indent=2), encoding='utf-8')
print('All stages verified; historical artifacts unchanged.', flush=True)
