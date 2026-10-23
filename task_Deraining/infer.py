"""Single-image inference from a trusted task checkpoint."""
import argparse
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor,to_pil_image
from runtime import select_device,load_model,restore_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='auto')
    a = p.parse_args()
    device = select_device(a.device)
    model,_ = load_model(a.checkpoint,device)
    with Image.open(a.input) as image:
        x = to_tensor(image.convert('RGB'))[None].to(device)
    with torch.inference_mode():
        y = restore_image(model,x).clamp(0,1)[0].cpu()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    to_pil_image(y).save(a.output)


if __name__ == '__main__':
    main()
