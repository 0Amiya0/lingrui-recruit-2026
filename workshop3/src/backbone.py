"""冻结的 ImageNet 预训练 ResNet 特征提取器。

输出 penultimate 层（去掉了最后的全连接）特征，默认 512 维。
backbone 从不接触 CIFAR 标签，保证 few-shot 设定的诚实性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Backbone(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = True, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        if name == "resnet18":
            factory, weights_cls, dim = models.resnet18, models.ResNet18_Weights, 512
        elif name == "resnet34":
            factory, weights_cls, dim = models.resnet34, models.ResNet34_Weights, 512
        elif name == "resnet50":
            factory, weights_cls, dim = models.resnet50, models.ResNet50_Weights, 2048
        else:
            raise ValueError(f"不支持的 backbone: {name}")

        try:
            weights = weights_cls.IMAGENET1K_V1 if pretrained else None
            model = factory(weights=weights)
        except Exception:  # 兼容旧版 torchvision 的 pretrained 参数
            model = factory(pretrained=pretrained)

        self.feature_dim = dim
        # 去掉最后的 avgpool 之后的 fc，保留到 global average pooling 前
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x).flatten(1)  # (B, dim)
        if self.normalize:
            f = F.normalize(f, dim=1)
        return f
