"""Written but not executed locally, per user's instruction."""
from pathlib import Path
import tempfile
import unittest
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from models import build_model
from dataset import DerainDataset, split_pairs
from runtime import restore_image
from utils import psnr
from losses import PerceptualLoss, EasyContrastiveLoss


class Tests(unittest.TestCase):
    def test_model_shapes_and_gradients(self):
        baseline = build_model('baseline')
        self.assertEqual(sum(isinstance(m,nn.Conv2d) for m in baseline.body.modules()),8)
        for name in ('baseline','derainnet','mymodel','unet'):
            model = build_model(name)
            x = torch.rand(1,3,32,40)
            y = model(x)
            self.assertEqual(y.shape, x.shape)
            y.square().mean().backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))
        model = build_model('mymodel')
        self.assertLess(sum(p.numel() for p in model.parameters()),4_000_000)
        self.assertEqual(restore_image(build_model('unet').eval(), torch.rand(1,3,35,41)).shape, (1,3,35,41))

    def test_psnr(self):
        x = torch.zeros(1,3,8,8)
        self.assertEqual(psnr(x,x), float('inf'))
        self.assertAlmostEqual(psnr(x, x+.1), 20, places=4)

    def test_synchronized_augmentation_and_group_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ('input','target'):
                (root/'train'/folder).mkdir(parents=True)
                for name in ('1_1.png','1_2.png','2_1.png','3_1.png'):
                    array = np.arange(48*48*3, dtype=np.uint8).reshape(48,48,3)
                    Image.fromarray(array).save(root/'train'/folder/name)
            train,val = split_pairs(root)
            self.assertFalse({a.stem.split('_')[0] for a,_ in train} & {a.stem.split('_')[0] for a,_ in val})
            ds = DerainDataset(pairs=train,image_size=32)
            random.seed(2)
            a,b,_ = ds[0]
            torch.testing.assert_close(a,b)

    def test_feature_losses_retain_prediction_gradient(self):
        # No downloads: actual loss behavior with a deterministic stand-in encoder.
        x = torch.rand(2,3,8,8,requires_grad=True)
        clean,rainy = torch.rand_like(x),torch.rand_like(x)
        for loss in (PerceptualLoss(nn.Identity()),EasyContrastiveLoss(nn.Identity()),EasyContrastiveLoss(nn.Identity(),mode='infonce')):
            value = loss(x,clean) if isinstance(loss,PerceptualLoss) else loss(rainy,x,clean)
            value.backward(retain_graph=True)
            self.assertTrue(torch.isfinite(value))
            self.assertIsNotNone(x.grad)

    def test_contrastive_direction(self):
        clean = torch.zeros(1,3,8,8)
        rainy = torch.zeros_like(clean)
        clean[:,0] = 1
        rainy[:,1] = 1
        for mode in ('cosine','infonce'):
            criterion = EasyContrastiveLoss(nn.Identity(),mode=mode)
            self.assertLess(criterion(rainy,clean,clean).item(),criterion(rainy,rainy,clean).item())


if __name__ == '__main__':
    unittest.main()
