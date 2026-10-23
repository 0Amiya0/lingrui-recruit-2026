from .baseline_net import BaselineNet
from .DerainNet import DerainNet
from .MyModel import MyModel


def build_model(name):
    if name == 'unet':
        from .unet import UNet
        return UNet()
    constructors = {'baseline': BaselineNet, 'derainnet': DerainNet, 'mymodel': MyModel}
    if name not in constructors:
        raise ValueError(f'unknown model: {name}')
    model = constructors[name]()
    if name == 'mymodel' and sum(p.numel() for p in model.parameters()) >= 4_000_000:
        raise ValueError('MyModel must have fewer than 4M parameters')
    return model
