import importlib.util
import ast
from pathlib import Path
import sys
import unittest
import tempfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nanoGPT'))
from model import GPT, GPTConfig


def load(name):
    tree = ast.parse((ROOT / (name + '.py')).read_text(encoding='utf-8'))
    expected = 'parse_log' if name == 'plot_loss' else 'generate_text'
    assert any(isinstance(node, ast.FunctionDef) and node.name == expected for node in tree.body), 'missing function: ' + expected
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ToolTests(unittest.TestCase):
    def test_log_parser(self):
        tool = load('plot_loss')
        self.assertTrue(hasattr(tool, 'parse_log'), 'missing import-safe parser')
        result = tool.parse_log('iter 0: loss 4.2e+0, time 1ms\nstep 0: train loss 4.1, val loss 4.3\nstep 3: train loss 2.0, val loss 2.2')
        self.assertEqual(result['iterations'][0], 4.2)
        self.assertEqual(result['evaluation'][3], (2.0, 2.2))
        with self.assertRaises(ValueError):
            tool.parse_log('no evaluation records')

    def test_sampling_reproducibility_and_validation(self):
        tool = load('generate_demo')
        self.assertTrue(hasattr(tool, 'generate_text'), 'missing reusable sampling function')
        model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=8, block_size=4, vocab_size=5, dropout=0))
        first = tool.generate_text(model, [1], max_new_tokens=8, seed=42)
        second = tool.generate_text(model, [1], max_new_tokens=8, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 9)
        for kwargs in ({'temperature': 0}, {'top_k': 0}, {'max_new_tokens': -1}):
            with self.assertRaises(ValueError):
                tool.generate_text(model, [1], **kwargs)
        with self.assertRaises(ValueError):
            tool.generate_text(model, [])

    def test_causal_shapes_and_weight_tying(self):
        torch.manual_seed(7)
        model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=8, block_size=4, vocab_size=5, dropout=0)).eval()
        x = torch.tensor([[1, 2, 3, 4]])
        y = torch.tensor([[2, 3, 4, 0]])
        changed = torch.tensor([[1, 2, 0, 0]])
        with torch.no_grad():
            logits, loss = model(x, y)
            alternate, _ = model(changed, y)
            final, _ = model(x)
        self.assertEqual(tuple(logits.shape), (1, 4, 5))
        self.assertEqual(tuple(final.shape), (1, 1, 5))
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(logits[:, :2], alternate[:, :2])
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)

    def test_checkpoint_round_trip_with_compiled_prefix(self):
        tool = load('generate_demo')
        config = GPTConfig(n_layer=1, n_head=2, n_embd=8, block_size=4, vocab_size=5, dropout=0)
        model = GPT(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ckpt.pt'
            torch.save({'model_args': vars(config), 'model': {'_orig_mod.' + k: v for k, v in model.state_dict().items()}}, path)
            restored, _ = tool.load_checkpoint(path)
            x = torch.tensor([[1, 2]])
            with torch.no_grad():
                torch.testing.assert_close(model(x)[0], restored(x)[0])

    def test_actual_training_batch_target_shift(self):
        # Execute only upstream get_batch, avoiding train.py's top-level training loop.
        source = ast.parse((ROOT / 'nanoGPT/train.py').read_text(encoding='utf-8'))
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'get_batch')
        with tempfile.TemporaryDirectory() as directory:
            sequence = np.arange(40, dtype=np.uint16)
            for split in ('train', 'val'):
                sequence.tofile(Path(directory) / (split + '.bin'))
            namespace = {'np': np, 'torch': torch, 'os': __import__('os'), 'data_dir': directory,
                         'batch_size': 2, 'block_size': 4, 'device_type': 'cpu', 'device': 'cpu'}
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'train.py:get_batch', 'exec'), namespace)
            x, y = namespace['get_batch']('train')
            self.assertEqual(tuple(x.shape), (2, 4))
            torch.testing.assert_close(y, x + 1)


if __name__ == '__main__':
    unittest.main()
