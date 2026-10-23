"""Count actual parameters and export a fixed, divisible-by-eight ONNX graph."""
import argparse
from pathlib import Path
import torch
from models import build_model
from runtime import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['baseline','unet','derainnet','mymodel'],default='mymodel')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,default=Path('exports/mymodel.onnx'))
    p.add_argument('--size',type=int,default=128)
    a = p.parse_args()
    if a.size < 32 or a.size % 8:
        p.error('size must be >=32 and divisible by 8')
    model = load_model(a.checkpoint)[0] if a.checkpoint else build_model(a.model).eval()
    print('Parameters:',sum(p.numel() for p in model.parameters()))
    if a.checkpoint is None:
        print('Architecture-only export with random weights, not a trained model.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,torch.zeros(1,3,a.size,a.size),str(a.output),input_names=['rainy'],output_names=['clean'],
                      opset_version=17,dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(str(a.output)))


if __name__ == '__main__':
    main()
