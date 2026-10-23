"""Models loaded once at startup; no downloads or inference on module import."""
import base64
import io
from pathlib import Path
import sys


def select_device(name):
    import torch
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('device must be auto/cpu/cuda/cuda:N')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; install CUDA-enabled PyTorch')
    return device


class Classifier:
    """ImageNet pretrained ResNet-18; first use may download weights on target host."""
    def __init__(self, device='auto'):
        from torchvision.models import resnet18, ResNet18_Weights
        self.device = select_device(device)
        weights = ResNet18_Weights.DEFAULT
        self.transform = weights.transforms()
        self.categories = weights.meta['categories']
        self.model = resnet18(weights=weights).to(self.device).eval()

    def predict(self, image):
        import torch
        with torch.inference_mode():
            probs = self.model(self.transform(image)[None].to(self.device)).softmax(1)[0]
            values, indices = probs.topk(5)
        return {'model': 'resnet18-imagenet', 'device': str(self.device), 'predictions': [
            {'label': self.categories[i], 'probability': float(p)} for p, i in zip(values.cpu().tolist(), indices.cpu().tolist())]}


class Derainer:
    def __init__(self, checkpoint, device='auto'):
        import torch
        root = Path(__file__).resolve().parents[1] / 'task_Deraining'
        sys.path.insert(0, str(root))
        from runtime import load_model, restore_image
        self.restore_image = restore_image
        self.device = select_device(device)
        self.model, self.metadata = load_model(checkpoint, self.device)

    def predict(self, image):
        import torch
        from torchvision.transforms.functional import to_tensor, to_pil_image
        with torch.inference_mode():
            result = self.restore_image(self.model, to_tensor(image)[None].to(self.device)).clamp(0, 1)
        buffer = io.BytesIO()
        to_pil_image(result[0].cpu()).save(buffer, format='PNG')
        return {'model': self.metadata['model'], 'device': str(self.device), 'width': image.width,
                'height': image.height, 'mime_type': 'image/png',
                'image_base64': base64.b64encode(buffer.getvalue()).decode('ascii')}
